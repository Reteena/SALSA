#!/usr/bin/env python3
"""
Main training script for SALSA with proper utterance-level data handling.
Supports both ERM and GroupDRO training with the correct data pipeline.
"""
import argparse
import os
import sys
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add SALSA to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.utterance_dataset import UtteranceDataset, collate_utterance_batch
from fusion.multimodal_fusion import create_multimodal_salsa, create_lexical_only_salsa, create_acoustic_only_salsa
try:
    from train.trainer import Trainer, FocalLoss, GroupDROLoss
except ImportError:
    from trainer import Trainer, FocalLoss, GroupDROLoss
from eval.metrics import MetricsCalculator


def setup_logging(experiment_name: str, output_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'{experiment_name}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(args) -> nn.Module:
    """Create model based on arguments."""
    model_kwargs = {
        'fusion_dim': args.fusion_dim,
        'num_classes': args.num_classes,
        'lora_rank': args.lora_rank,
        'use_cross_attention': args.use_cross_attention,
        'aggregation_type': args.aggregation_type,
        'dropout': args.dropout
    }
    
    if args.model_type == 'multimodal':
        model = create_multimodal_salsa(
            acoustic_dim=args.acoustic_dim,
            lexical_dim=args.lexical_dim,
            **model_kwargs
        )
    elif args.model_type == 'lexical_only':
        # Remove conflicting parameters for lexical-only model
        lexical_kwargs = {k: v for k, v in model_kwargs.items() if k != 'use_cross_attention'}
        model = create_lexical_only_salsa(
            lexical_dim=args.lexical_dim,
            **lexical_kwargs
        )
    elif args.model_type == 'acoustic_only':
        # Remove conflicting parameters for acoustic-only model  
        acoustic_kwargs = {k: v for k, v in model_kwargs.items() if k != 'use_cross_attention'}
        model = create_acoustic_only_salsa(
            acoustic_dim=args.acoustic_dim,
            **acoustic_kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model


def create_data_loaders(args) -> tuple:
    """Create train, validation, and test data loaders."""
    dataset_kwargs = {
        'max_utterances_per_recording': args.max_utterances_per_recording,
        'include_audio': args.model_type in ['multimodal', 'acoustic_only'],
        'include_lexical': args.model_type in ['multimodal', 'lexical_only'],
        'normalize_features': args.normalize_features,
        'filter_min_utterances': args.filter_min_utterances
    }
    
    # Create datasets
    train_dataset = UtteranceDataset(args.train_csv, **dataset_kwargs)
    val_dataset = UtteranceDataset(args.val_csv, **dataset_kwargs)
    test_dataset = UtteranceDataset(args.test_csv, **dataset_kwargs)
    
    # Print dataset stats
    print(f"Train dataset: {len(train_dataset)} recordings, {train_dataset.total_utterances} utterances")
    print(f"Val dataset: {len(val_dataset)} recordings, {val_dataset.total_utterances} utterances")
    print(f"Test dataset: {len(test_dataset)} recordings, {test_dataset.total_utterances} utterances")
    
    # Create data loaders
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_utterance_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, train_dataset.get_class_weights()


def create_loss_function(args, class_weights=None):
    """Create loss function based on arguments."""
    if args.loss_type == 'focal':
        return FocalLoss(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing
        )
    elif args.loss_type == 'cross_entropy':
        weight = None
        if class_weights and args.use_class_weights:
            weight = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    elif args.loss_type == 'group_dro':
        return GroupDROLoss(
            num_groups=args.num_groups,
            group_weights_lr=args.group_weights_lr
        )
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")


def train_step(model, batch, criterion, device, model_type='multimodal'):
    """Single training step."""
    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    labels = batch['labels']
    attention_mask = batch.get('attention_mask', None)
    
    # Forward pass based on model type
    if model_type == 'multimodal':
        logits = model(
            lexical_features=batch['lexical_features'],
            acoustic_features=batch.get('acoustic_features', None),
            attention_mask=attention_mask
        )
    elif model_type == 'lexical_only':
        logits = model(
            lexical_features=batch['lexical_features'],
            attention_mask=attention_mask
        )
    elif model_type == 'acoustic_only':
        logits = model(
            acoustic_features=batch['acoustic_features'],
            attention_mask=attention_mask
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Compute loss
    loss = criterion(logits, labels)
    
    return logits, loss


def validate_model(model, val_loader, criterion, device, model_type='multimodal'):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_recording_ids = []
    
    with torch.no_grad():
        for batch in val_loader:
            logits, loss = train_step(model, batch, criterion, device, model_type)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_recording_ids.extend(batch['recording_ids'])
    
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.compute_classification_metrics(
        predictions=torch.tensor(all_predictions, dtype=torch.long),
        labels=torch.tensor(all_labels, dtype=torch.long)
    )
    
    return avg_loss, metrics, all_predictions, all_labels, all_recording_ids


def main():
    parser = argparse.ArgumentParser(description="Train SALSA model with utterance-level features")
    
    # Data arguments
    parser.add_argument('--train_csv', required=True, help='Path to training CSV file')
    parser.add_argument('--val_csv', required=True, help='Path to validation CSV file')
    parser.add_argument('--test_csv', required=True, help='Path to test CSV file')
    
    # Model arguments
    parser.add_argument('--model_type', choices=['multimodal', 'lexical_only', 'acoustic_only'], 
                        default='multimodal', help='Type of model to train')
    parser.add_argument('--acoustic_dim', type=int, default=768, help='Acoustic feature dimension')
    parser.add_argument('--lexical_dim', type=int, default=400, help='Lexical feature dimension (384 emb + 16 handcrafted)')
    parser.add_argument('--fusion_dim', type=int, default=512, help='Fusion layer dimension')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--use_cross_attention', action='store_true', default=True, help='Use cross-attention')
    parser.add_argument('--aggregation_type', choices=['attention', 'mean', 'max', 'lstm', 'transformer'],
                        default='attention', help='Utterance aggregation method')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Early stopping minimum delta')
    
    # Loss function arguments
    parser.add_argument('--loss_type', choices=['cross_entropy', 'focal', 'group_dro'], 
                        default='focal', help='Loss function type')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for imbalanced data')
    
    # GroupDRO arguments
    parser.add_argument('--use_group_dro', action='store_true', help='Use Group DRO training')
    parser.add_argument('--num_groups', type=int, default=2, help='Number of groups for GroupDRO')
    parser.add_argument('--group_weights_lr', type=float, default=0.01, help='Group weights learning rate')
    
    # Data processing arguments
    parser.add_argument('--max_utterances_per_recording', type=int, default=100, 
                        help='Maximum utterances per recording')
    parser.add_argument('--normalize_features', action='store_true', default=True,
                        help='Normalize features')
    parser.add_argument('--filter_min_utterances', type=int, default=1,
                        help='Minimum utterances per recording')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', required=True, help='Experiment name')
    parser.add_argument('--output_dir', default='./experiments', help='Output directory')
    parser.add_argument('--save_model', action='store_true', default=True, help='Save trained model')
    
    args = parser.parse_args()
    
    # Setup
    set_random_seeds(args.seed)
    
    # Create output directory
    exp_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.experiment_name, exp_dir)
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader, class_weights = create_data_loaders(args)
        logger.info(f"Class weights: {class_weights}")
        
        # Create model
        model = create_model(args)
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")
        
        # Create loss function
        if args.use_group_dro:
            criterion = GroupDROLoss(num_groups=args.num_groups, group_weights_lr=args.group_weights_lr)
        else:
            criterion = create_loss_function(args, class_weights)
        criterion = criterion.to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=args.patience // 2,
            verbose=True
        )
        
        # Training loop
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(args.max_epochs):
            logger.info(f"Epoch {epoch+1}/{args.max_epochs}")
            
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                logits, loss = train_step(model, batch, criterion, device, args.model_type)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_loss, val_metrics, _, _, _ = validate_model(model, val_loader, criterion, device, args.model_type)
            val_f1 = val_metrics['macro_f1']
            
            # Learning rate scheduling
            scheduler.step(val_f1)
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1 + args.min_delta:
                best_val_f1 = val_f1
                patience_counter = 0
                
                if args.save_model:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_val_f1': best_val_f1,
                        'args': vars(args)
                    }, os.path.join(exp_dir, 'best_model.pth'))
                    
                logger.info(f"New best validation F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final test evaluation
        logger.info("Evaluating on test set...")
        
        # Load best model
        if args.save_model:
            checkpoint = torch.load(os.path.join(exp_dir, 'best_model.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_metrics, test_preds, test_labels, test_ids = validate_model(
            model, test_loader, criterion, device, args.model_type
        )
        
        logger.info(f"Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save results
        results = {
            'experiment_name': args.experiment_name,
            'args': vars(args),
            'best_val_f1': best_val_f1,
            'test_metrics': test_metrics,
            'model_params': {
                'total': total_params,
                'trainable': trainable_params
            }
        }
        
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Experiment completed. Results saved to {exp_dir}")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()