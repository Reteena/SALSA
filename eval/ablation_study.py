"""
Comprehensive ablation study script for SALSA.
Tests different model configurations, hyperparameters, and design choices.
"""
import os
import sys
import argparse
import logging
import torch
import torch.utils.data
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json
import itertools
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from train.trainer import Trainer, TrainingConfig
from eval.metrics import compute_classification_metrics
from data.utterance_dataset import UtteranceDataset, collate_utterance_batch
from fusion.multimodal_fusion import MultimodalFusion
from eval.evaluate import RecordingLevelEvaluator


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    model_type: str  # 'multimodal', 'acoustic_only', 'lexical_only'
    use_group_dro: bool
    lora_rank: int
    num_fusion_layers: int
    dropout: float
    learning_rate: float
    focal_gamma: float
    batch_size: int
    description: str = ""


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


class AblationStudy:
    """Comprehensive ablation study runner."""
    
    def __init__(self, 
                 base_config: TrainingConfig,
                 device: torch.device,
                 data_dir: str,
                 train_manifest: str,
                 val_manifest: str,
                 test_manifest: str = None):
        
        self.base_config = base_config
        self.device = device
        self.data_dir = data_dir
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.test_manifest = test_manifest
        
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
        # Create data loaders once
        self.train_dataset = UtteranceDataset(train_manifest)
        self.val_dataset = UtteranceDataset(val_manifest)
        self.test_dataset = UtteranceDataset(test_manifest) if test_manifest else None
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=base_config.batch_size,
            shuffle=True,
            collate_fn=collate_utterance_batch,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=base_config.batch_size,
            shuffle=False,
            collate_fn=collate_utterance_batch,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = None
        if self.test_dataset:
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=base_config.batch_size,
                shuffle=False,
                collate_fn=collate_utterance_batch,
                num_workers=4,
                pin_memory=True
            )
    
    def run_single_ablation(self, ablation_config: AblationConfig) -> Dict[str, Any]:
        """Run a single ablation experiment."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running ablation: {ablation_config.name}")
        self.logger.info(f"Description: {ablation_config.description}")
        self.logger.info(f"{'='*60}")
        
        # Create model based on ablation config
        model = self._create_ablation_model(ablation_config)
        
        # Create training config for this ablation
        train_config = self._create_ablation_training_config(ablation_config)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=train_config,
            device=self.device,
            experiment_name=f"ablation_{ablation_config.name}",
            log_dir=f"./logs/ablations"
        )
        
        # Train model
        training_history = trainer.fit(self.train_loader, self.val_loader)
        
        # Evaluate on validation set
        evaluator = RecordingLevelEvaluator(model, self.device, train_config)
        val_result = evaluator.evaluate_dataset(self.val_dataset, batch_size=train_config.batch_size)
        val_scores = val_result['metrics']
        
        # Evaluate on test set if available
        test_scores = None
        if self.test_dataset is not None:
            test_result = evaluator.evaluate_dataset(self.test_dataset, batch_size=train_config.batch_size)
            test_scores = test_result['metrics']
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            'ablation_config': asdict(ablation_config),
            'val_metrics': val_scores,
            'test_metrics': test_scores,
            'training_history': training_history,
            'model_params': {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'trainable_percent': 100 * trainable_params / total_params
            }
        }
        
        # Log key results
        self.logger.info(f"Results for {ablation_config.name}:")
        self.logger.info(f"  Val Macro F1: {val_scores['macro_f1']:.4f}")
        self.logger.info(f"  Val AUROC: {val_scores['auroc']:.4f}")
        self.logger.info(f"  Val ECE: {val_scores['ece']:.4f}")
        self.logger.info(f"  Trainable Params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        if test_scores:
            self.logger.info(f"  Test Macro F1: {test_scores['macro_f1']:.4f}")
            self.logger.info(f"  Test AUROC: {test_scores['auroc']:.4f}")
        
        if 'worst_group_f1' in val_scores:
            self.logger.info(f"  Val Worst Group F1: {val_scores['worst_group_f1']:.4f}")
        
        return results
    
    def _create_ablation_model(self, config: AblationConfig) -> torch.nn.Module:
        """Create model for ablation study."""
        if config.model_type == "multimodal":
            return MultimodalFusion(
                lexical_dim=self.base_config.lexical_dim,
                num_classes=self.base_config.num_classes,
                hidden_dim=self.base_config.hidden_dim,
                num_fusion_layers=config.num_fusion_layers,
                dropout=config.dropout,
                aggregation_strategy="attention"
            )
        elif config.model_type == "lexical_only":
            # For lexical only, we use a simplified version of the fusion model
            return MultimodalFusion(
                lexical_dim=self.base_config.lexical_dim,
                num_classes=self.base_config.num_classes,
                hidden_dim=self.base_config.hidden_dim,
                num_fusion_layers=1,
                dropout=config.dropout,
                aggregation_strategy="attention",
                modality_mode="lexical_only"
            )
        else:
            raise ValueError(f"Unknown model type: {config.model_type}. Only 'multimodal' and 'lexical_only' supported for utterance-level evaluation.")
    
    def _create_ablation_training_config(self, config: AblationConfig) -> TrainingConfig:
        """Create training config for ablation."""
        train_config = TrainingConfig()
        
        # Copy base config
        for key, value in self.base_config.__dict__.items():
            setattr(train_config, key, value)
        
        # Override with ablation-specific settings
        train_config.use_group_dro = config.use_group_dro
        train_config.lora_rank = config.lora_rank
        train_config.num_fusion_layers = config.num_fusion_layers
        train_config.dropout = config.dropout
        train_config.learning_rate = config.learning_rate
        train_config.focal_gamma = config.focal_gamma
        train_config.batch_size = config.batch_size
        
        # Shorter training for ablations
        train_config.max_epochs = min(train_config.max_epochs, 30)
        train_config.patience = min(train_config.patience, 10)
        
        return train_config
    
    def run_modality_ablation(self) -> Dict[str, Any]:
        """Test different modality combinations."""
        self.logger.info("\n" + "="*50)
        self.logger.info("MODALITY ABLATION STUDY")
        self.logger.info("="*50)
        
        ablation_configs = [
            AblationConfig(
                name="acoustic_only",
                model_type="acoustic_only",
                use_group_dro=self.base_config.use_group_dro,
                lora_rank=self.base_config.lora_rank,
                num_fusion_layers=0,  # Not applicable
                dropout=self.base_config.dropout,
                learning_rate=self.base_config.learning_rate,
                focal_gamma=self.base_config.focal_gamma,
                batch_size=self.base_config.batch_size,
                description="Acoustic features only (frozen WavLM + LoRA)"
            ),
            AblationConfig(
                name="lexical_only",
                model_type="lexical_only",
                use_group_dro=self.base_config.use_group_dro,
                lora_rank=0,  # Not applicable
                num_fusion_layers=0,  # Not applicable
                dropout=self.base_config.dropout,
                learning_rate=self.base_config.learning_rate,
                focal_gamma=self.base_config.focal_gamma,
                batch_size=self.base_config.batch_size,
                description="Lexical features only (ASR + linguistic features + MiniLM)"
            ),
            AblationConfig(
                name="multimodal_fusion",
                model_type="multimodal",
                use_group_dro=self.base_config.use_group_dro,
                lora_rank=self.base_config.lora_rank,
                num_fusion_layers=self.base_config.num_fusion_layers,
                dropout=self.base_config.dropout,
                learning_rate=self.base_config.learning_rate,
                focal_gamma=self.base_config.focal_gamma,
                batch_size=self.base_config.batch_size,
                description="Full multimodal fusion with gated cross-attention"
            )
        ]
        
        results = {}
        for config in ablation_configs:
            results[config.name] = self.run_single_ablation(config)
        
        return results
    
    def run_group_dro_ablation(self) -> Dict[str, Any]:
        """Test GroupDRO vs ERM."""
        self.logger.info("\n" + "="*50)
        self.logger.info("GROUP DRO vs ERM ABLATION")
        self.logger.info("="*50)
        
        ablation_configs = [
            AblationConfig(
                name="erm_training",
                model_type="multimodal",
                use_group_dro=False,
                lora_rank=self.base_config.lora_rank,
                num_fusion_layers=self.base_config.num_fusion_layers,
                dropout=self.base_config.dropout,
                learning_rate=self.base_config.learning_rate,
                focal_gamma=self.base_config.focal_gamma,
                batch_size=self.base_config.batch_size,
                description="Standard ERM training (no GroupDRO)"
            ),
            AblationConfig(
                name="group_dro_training",
                model_type="multimodal",
                use_group_dro=True,
                lora_rank=self.base_config.lora_rank,
                num_fusion_layers=self.base_config.num_fusion_layers,
                dropout=self.base_config.dropout,
                learning_rate=self.base_config.learning_rate,
                focal_gamma=self.base_config.focal_gamma,
                batch_size=self.base_config.batch_size,
                description="GroupDRO training for worst-group optimization"
            )
        ]
        
        results = {}
        for config in ablation_configs:
            results[config.name] = self.run_single_ablation(config)
        
        return results
    
    def run_lora_rank_ablation(self) -> Dict[str, Any]:
        """Test different LoRA ranks."""
        self.logger.info("\n" + "="*50)
        self.logger.info("LORA RANK ABLATION")
        self.logger.info("="*50)
        
        lora_ranks = [0, 4, 8, 16, 32]
        results = {}
        
        for rank in lora_ranks:
            config = AblationConfig(
                name=f"lora_rank_{rank}",
                model_type="multimodal",
                use_group_dro=self.base_config.use_group_dro,
                lora_rank=rank,
                num_fusion_layers=self.base_config.num_fusion_layers,
                dropout=self.base_config.dropout,
                learning_rate=self.base_config.learning_rate,
                focal_gamma=self.base_config.focal_gamma,
                batch_size=self.base_config.batch_size,
                description=f"LoRA rank {rank} ({'no LoRA' if rank == 0 else 'LoRA enabled'})"
            )
            
            results[config.name] = self.run_single_ablation(config)
        
        return results
    
    def run_fusion_layers_ablation(self) -> Dict[str, Any]:
        """Test different numbers of fusion layers."""
        self.logger.info("\n" + "="*50)
        self.logger.info("FUSION LAYERS ABLATION")
        self.logger.info("="*50)
        
        fusion_layers = [1, 2, 3, 4]
        results = {}
        
        for num_layers in fusion_layers:
            config = AblationConfig(
                name=f"fusion_layers_{num_layers}",
                model_type="multimodal",
                use_group_dro=self.base_config.use_group_dro,
                lora_rank=self.base_config.lora_rank,
                num_fusion_layers=num_layers,
                dropout=self.base_config.dropout,
                learning_rate=self.base_config.learning_rate,
                focal_gamma=self.base_config.focal_gamma,
                batch_size=self.base_config.batch_size,
                description=f"Multimodal fusion with {num_layers} cross-attention layers"
            )
            
            results[config.name] = self.run_single_ablation(config)
        
        return results
    
    def run_hyperparameter_ablation(self) -> Dict[str, Any]:
        """Test different hyperparameters."""
        self.logger.info("\n" + "="*50)
        self.logger.info("HYPERPARAMETER ABLATION")
        self.logger.info("="*50)
        
        # Learning rate ablation
        learning_rates = [5e-5, 1e-4, 2e-4, 5e-4]
        dropout_rates = [0.1, 0.3, 0.5]
        focal_gammas = [1.0, 2.0, 3.0]
        
        results = {}
        
        # Learning rate sweep
        for lr in learning_rates:
            config = AblationConfig(
                name=f"lr_{lr:.0e}",
                model_type="multimodal",
                use_group_dro=self.base_config.use_group_dro,
                lora_rank=self.base_config.lora_rank,
                num_fusion_layers=self.base_config.num_fusion_layers,
                dropout=self.base_config.dropout,
                learning_rate=lr,
                focal_gamma=self.base_config.focal_gamma,
                batch_size=self.base_config.batch_size,
                description=f"Learning rate {lr:.0e}"
            )
            results[config.name] = self.run_single_ablation(config)
        
        # Dropout sweep
        for dropout in dropout_rates:
            config = AblationConfig(
                name=f"dropout_{dropout}",
                model_type="multimodal",
                use_group_dro=self.base_config.use_group_dro,
                lora_rank=self.base_config.lora_rank,
                num_fusion_layers=self.base_config.num_fusion_layers,
                dropout=dropout,
                learning_rate=self.base_config.learning_rate,
                focal_gamma=self.base_config.focal_gamma,
                batch_size=self.base_config.batch_size,
                description=f"Dropout rate {dropout}"
            )
            results[config.name] = self.run_single_ablation(config)
        
        # Focal loss gamma sweep
        for gamma in focal_gammas:
            config = AblationConfig(
                name=f"focal_gamma_{gamma}",
                model_type="multimodal",
                use_group_dro=self.base_config.use_group_dro,
                lora_rank=self.base_config.lora_rank,
                num_fusion_layers=self.base_config.num_fusion_layers,
                dropout=self.base_config.dropout,
                learning_rate=self.base_config.learning_rate,
                focal_gamma=gamma,
                batch_size=self.base_config.batch_size,
                description=f"Focal loss gamma {gamma}"
            )
            results[config.name] = self.run_single_ablation(config)
        
        return results
    
    def run_complete_ablation_study(self) -> Dict[str, Any]:
        """Run complete ablation study."""
        self.logger.info("Starting comprehensive ablation study...")
        
        all_results = {}
        
        # Run different ablation categories
        studies = [
            ("modality", self.run_modality_ablation),
            ("group_dro", self.run_group_dro_ablation),
            ("lora_rank", self.run_lora_rank_ablation),
            ("fusion_layers", self.run_fusion_layers_ablation),
            ("hyperparameters", self.run_hyperparameter_ablation)
        ]
        
        for study_name, study_func in studies:
            self.logger.info(f"\nStarting {study_name} ablation...")
            results = study_func()
            all_results[study_name] = results
            self.logger.info(f"Completed {study_name} ablation.")
        
        return all_results
    
    def generate_ablation_report(self, results: Dict[str, Any], output_dir: str):
        """Generate comprehensive ablation study report."""
        self.logger.info("Generating ablation study report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Collect all results into a flat structure
        flat_results = []
        
        for study_category, study_results in results.items():
            for experiment_name, experiment_results in study_results.items():
                row = {
                    'study_category': study_category,
                    'experiment_name': experiment_name,
                    'model_type': experiment_results['ablation_config']['model_type'],
                    'use_group_dro': experiment_results['ablation_config']['use_group_dro'],
                    'lora_rank': experiment_results['ablation_config']['lora_rank'],
                    'num_fusion_layers': experiment_results['ablation_config']['num_fusion_layers'],
                    'dropout': experiment_results['ablation_config']['dropout'],
                    'learning_rate': experiment_results['ablation_config']['learning_rate'],
                    'focal_gamma': experiment_results['ablation_config']['focal_gamma'],
                    'description': experiment_results['ablation_config']['description'],
                    'total_params': experiment_results['model_params']['total_params'],
                    'trainable_params': experiment_results['model_params']['trainable_params'],
                    'trainable_percent': experiment_results['model_params']['trainable_percent']
                }
                
                # Add validation metrics
                val_metrics = experiment_results['val_metrics']
                for metric_name, metric_value in val_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        row[f'val_{metric_name}'] = metric_value
                
                # Add test metrics if available
                if experiment_results['test_metrics']:
                    test_metrics = experiment_results['test_metrics']
                    for metric_name, metric_value in test_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            row[f'test_{metric_name}'] = metric_value
                
                flat_results.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(flat_results)
        df.to_csv(output_path / "ablation_results.csv", index=False)
        
        # Generate summary report
        with open(output_path / "ablation_summary.txt", 'w') as f:
            f.write("SALSA Ablation Study Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Best performing configurations
            f.write("Best Performing Configurations:\n")
            f.write("-" * 40 + "\n")
            
            # Sort by validation macro F1
            df_sorted = df.sort_values('val_macro_f1', ascending=False)
            
            f.write("Top 5 by Validation Macro F1:\n")
            for i, (_, row) in enumerate(df_sorted.head(5).iterrows()):
                f.write(f"{i+1}. {row['experiment_name']} "
                       f"(F1: {row['val_macro_f1']:.4f}, "
                       f"AUROC: {row['val_auroc']:.4f}, "
                       f"Params: {row['trainable_params']:,})\n")
                f.write(f"   {row['description']}\n")
            
            f.write("\n")
            
            # Category summaries
            for category in df['study_category'].unique():
                f.write(f"{category.upper()} ABLATION RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                category_df = df[df['study_category'] == category]
                category_sorted = category_df.sort_values('val_macro_f1', ascending=False)
                
                for _, row in category_sorted.iterrows():
                    f.write(f"{row['experiment_name']}: "
                           f"F1={row['val_macro_f1']:.4f}, "
                           f"AUROC={row['val_auroc']:.4f}, "
                           f"ECE={row['val_ece']:.4f}\n")
                f.write("\n")
        
        # Save detailed results as JSON
        with open(output_path / "ablation_detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Ablation report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SALSA Ablation Study')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_manifest', type=str, required=True)
    parser.add_argument('--val_manifest', type=str, required=True)
    parser.add_argument('--test_manifest', type=str, default=None)
    
    # Ablation study settings
    parser.add_argument('--studies', nargs='+', 
                       choices=['modality', 'group_dro', 'lora_rank', 'fusion_layers', 'hyperparameters', 'all'],
                       default=['all'],
                       help='Which ablation studies to run')
    
    # Base model configuration
    parser.add_argument('--acoustic_dim', type=int, default=768)
    parser.add_argument('--lexical_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_fusion_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lora_rank', type=int, default=8)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--use_group_dro', action='store_true')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./ablation_results')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.experiment_name, args.output_dir)
    
    logger.info(f"Starting ablation study: {args.experiment_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Studies to run: {args.studies}")
    
    # Create base config
    base_config = TrainingConfig()
    base_config.acoustic_dim = args.acoustic_dim
    base_config.lexical_dim = args.lexical_dim
    base_config.hidden_dim = args.hidden_dim
    base_config.num_fusion_layers = args.num_fusion_layers
    base_config.dropout = args.dropout
    base_config.lora_rank = args.lora_rank
    base_config.batch_size = args.batch_size
    base_config.learning_rate = args.learning_rate
    base_config.max_epochs = args.max_epochs
    base_config.patience = args.patience
    base_config.use_group_dro = args.use_group_dro
    base_config.focal_gamma = args.focal_gamma
    
    # Create ablation study runner
    study = AblationStudy(
        base_config=base_config,
        device=device,
        data_dir=args.data_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        test_manifest=args.test_manifest
    )
    
    # Run ablation studies
    if 'all' in args.studies:
        results = study.run_complete_ablation_study()
    else:
        results = {}
        study_methods = {
            'modality': study.run_modality_ablation,
            'group_dro': study.run_group_dro_ablation,
            'lora_rank': study.run_lora_rank_ablation,
            'fusion_layers': study.run_fusion_layers_ablation,
            'hyperparameters': study.run_hyperparameter_ablation
        }
        
        for study_name in args.studies:
            if study_name in study_methods:
                results[study_name] = study_methods[study_name]()
    
    # Generate report
    study.generate_ablation_report(results, args.output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("ABLATION STUDY COMPLETED")
    logger.info("="*60)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("Check ablation_results.csv and ablation_summary.txt for detailed results")


if __name__ == "__main__":
    main()