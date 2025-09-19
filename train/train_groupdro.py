"""
Main training script for SALSA with GroupDRO.
Implements Group Distributionally Robust Optimization for better generalization.
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from fusion.multimodal_fusion import MultimodalFusion, create_lexical_only_salsa, create_acoustic_only_salsa, create_multimodal_salsa
try:
    from train.trainer import Trainer, TrainingConfig
except ImportError:
    from trainer import Trainer, TrainingConfig
from eval.metrics import compute_classification_metrics
from data.utterance_dataset import UtteranceDataset, collate_utterance_batch


def setup_logging(experiment_name: str, log_dir: str = "./logs"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/{experiment_name}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def create_model(config: TrainingConfig, model_type: str = "multimodal") -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Training configuration
        model_type: 'multimodal', 'acoustic_only', or 'lexical_only'
    """
    if model_type == "multimodal":
        model = create_multimodal_salsa(
            acoustic_dim=config.acoustic_dim,
            lexical_dim=config.lexical_dim,
            fusion_dim=config.fusion_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
    elif model_type == "acoustic_only":
        model = create_acoustic_only_salsa(
            acoustic_dim=config.acoustic_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
    elif model_type == "lexical_only":
        model = create_lexical_only_salsa(
            lexical_dim=config.lexical_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def analyze_group_distribution(data_loader):
    """Analyze the distribution of groups in the dataset."""
    group_counts = {}
    label_counts = {}
    group_label_counts = {}
    
    for batch in data_loader:
        group_ids = batch['group_ids'].numpy()
        labels = batch['labels'].numpy()
        
        for group_id, label in zip(group_ids, labels):
            group_id = int(group_id)
            label = int(label)
            
            # Count groups
            group_counts[group_id] = group_counts.get(group_id, 0) + 1
            
            # Count labels
            label_counts[label] = label_counts.get(label, 0) + 1
            
            # Count group-label combinations
            key = (group_id, label)
            group_label_counts[key] = group_label_counts.get(key, 0) + 1
    
    return group_counts, label_counts, group_label_counts


def main():
    parser = argparse.ArgumentParser(description='Train SALSA model with GroupDRO')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Training CSV file with utterance-level features')
    parser.add_argument('--val_csv', type=str, required=True,
                       help='Validation CSV file with utterance-level features')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='multimodal',
                       choices=['multimodal', 'acoustic_only', 'lexical_only'],
                       help='Type of model to train')
    parser.add_argument('--acoustic_dim', type=int, default=768,
                       help='Acoustic feature dimension')
    parser.add_argument('--lexical_dim', type=int, default=512,
                       help='Lexical feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=768,
                       help='Hidden dimension for fusion')
    parser.add_argument('--num_fusion_layers', type=int, default=2,
                       help='Number of fusion layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # LoRA arguments  
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    
    # GroupDRO specific arguments
    parser.add_argument('--group_weights_step_size', type=float, default=0.01,
                       help='Group weights step size for GroupDRO')
    parser.add_argument('--group_dro_alpha', type=float, default=0.1,
                       help='EMA alpha for group loss tracking')
    
    # Loss function arguments
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                       help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                       help='Label smoothing epsilon')
    
    # Scheduler arguments
    parser.add_argument('--scheduler_type', type=str, default='plateau',
                       choices=['plateau', 'cosine'],
                       help='Learning rate scheduler type')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                       help='Scheduler patience for plateau')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                       help='Factor for plateau scheduler')
    
    # Logging and saving
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Experiment name for logging')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints')
    
    # Evaluation arguments
    parser.add_argument('--eval_interval', type=int, default=1,
                       help='Evaluation interval in epochs')
    parser.add_argument('--bootstrap_samples', type=int, default=1000,
                       help='Number of bootstrap samples for confidence intervals')
    
    # Hardware
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detect if None')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Setup logging
    logger = setup_logging(args.experiment_name, args.log_dir)
    logger.info(f"Starting GroupDRO experiment: {args.experiment_name}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create training config
    config = TrainingConfig()
    config.acoustic_dim = args.acoustic_dim
    config.lexical_dim = args.lexical_dim
    config.hidden_dim = args.hidden_dim
    config.num_fusion_layers = args.num_fusion_layers
    config.dropout = args.dropout
    config.lora_rank = args.lora_rank
    config.lora_alpha = args.lora_alpha
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.max_epochs = args.max_epochs
    config.patience = args.patience
    config.gradient_clip = args.gradient_clip
    config.focal_alpha = args.focal_alpha
    config.focal_gamma = args.focal_gamma
    config.label_smoothing = args.label_smoothing
    config.use_group_dro = True  # GroupDRO training
    config.group_weights_step_size = args.group_weights_step_size
    config.scheduler_type = args.scheduler_type
    config.scheduler_patience = args.scheduler_patience
    config.scheduler_factor = args.scheduler_factor
    config.eval_interval = args.eval_interval
    config.bootstrap_samples = args.bootstrap_samples
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataset = UtteranceDataset(args.train_csv)
    val_dataset = UtteranceDataset(args.val_csv)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_utterance_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_utterance_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Analyze group distribution
    logger.info("Analyzing group distribution...")
    train_group_counts, train_label_counts, train_group_label_counts = analyze_group_distribution(train_loader)
    val_group_counts, val_label_counts, val_group_label_counts = analyze_group_distribution(val_loader)
    
    logger.info("Training set group distribution:")
    for group_id, count in sorted(train_group_counts.items()):
        logger.info(f"  Group {group_id}: {count} samples")
    
    logger.info("Training set label distribution:")
    for label, count in sorted(train_label_counts.items()):
        label_name = "Healthy" if label == 0 else "Dementia"
        logger.info(f"  {label_name}: {count} samples")
    
    logger.info("Training set group-label distribution:")
    for (group_id, label), count in sorted(train_group_label_counts.items()):
        label_name = "Healthy" if label == 0 else "Dementia"
        logger.info(f"  Group {group_id} - {label_name}: {count} samples")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = create_model(config, args.model_type)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        experiment_name=args.experiment_name,
        log_dir=args.log_dir
    )
    
    # Start training
    logger.info("Starting GroupDRO training...")
    logger.info("GroupDRO will adaptively weight groups to optimize worst-group performance")
    
    training_history = trainer.fit(train_loader, val_loader)
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_metrics = trainer.validate(val_loader)
    
    final_scores = compute_classification_metrics(
        final_metrics['predictions'],
        final_metrics['labels'],
        final_metrics['group_ids'],
        num_bootstrap=config.bootstrap_samples
    )
    
    logger.info("Final Results:")
    logger.info(f"  Macro F1: {final_scores['macro_f1']:.4f} "
               f"({final_scores['macro_f1_ci'][0]:.4f}-{final_scores['macro_f1_ci'][1]:.4f})")
    logger.info(f"  AUROC: {final_scores['auroc']:.4f} "
               f"({final_scores['auroc_ci'][0]:.4f}-{final_scores['auroc_ci'][1]:.4f})")
    logger.info(f"  Balanced Accuracy: {final_scores['balanced_accuracy']:.4f}")
    logger.info(f"  ECE: {final_scores['ece']:.4f}")
    
    # GroupDRO specific metrics
    if 'worst_group_f1' in final_scores:
        logger.info(f"  Worst Group F1: {final_scores['worst_group_f1']:.4f}")
        logger.info(f"  Best Group F1: {final_scores['best_group_f1']:.4f}")
        logger.info(f"  F1 Group Gap: {final_scores['f1_group_gap']:.4f}")
    
    if 'worst_group_auroc' in final_scores:
        logger.info(f"  Worst Group AUROC: {final_scores['worst_group_auroc']:.4f}")
        logger.info(f"  Best Group AUROC: {final_scores['best_group_auroc']:.4f}")
        logger.info(f"  AUROC Group Gap: {final_scores['auroc_group_gap']:.4f}")
    
    # Log final group weights from GroupDRO
    if hasattr(trainer.loss_fn, 'group_weights'):
        group_info = trainer.loss_fn.get_group_info()
        logger.info("Final GroupDRO weights:")
        for i, weight in enumerate(group_info['group_weights']):
            logger.info(f"  Group {i}: {weight:.4f}")
    
    # Save results
    results_file = os.path.join(args.log_dir, f"{args.experiment_name}_results.pt")
    torch.save({
        'config': config,
        'final_metrics': final_scores,
        'training_history': training_history,
        'group_distribution': {
            'train_group_counts': train_group_counts,
            'train_label_counts': train_label_counts,
            'train_group_label_counts': train_group_label_counts,
            'val_group_counts': val_group_counts,
            'val_label_counts': val_label_counts,
            'val_group_label_counts': val_group_label_counts
        },
        'final_group_info': trainer.loss_fn.get_group_info() if hasattr(trainer.loss_fn, 'get_group_info') else None,
        'args': vars(args)
    }, results_file)
    
    logger.info(f"Results saved to {results_file}")
    logger.info("GroupDRO training completed successfully!")
    logger.info("GroupDRO optimizes for worst-group performance, which should improve generalization.")


if __name__ == "__main__":
    main()