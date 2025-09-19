"""
Leave-One-Corpus-Out (LOCO) evaluation script for SALSA.
Tests generalization across different datasets/corpora.
"""
import os
import sys
import argparse
import logging
import itertools
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from train.trainer import Trainer, TrainingConfig
from eval.metrics import compute_classification_metrics, VisualizationUtils
from data.utterance_dataset import UtteranceDataset, collate_utterance_batch, create_data_loaders
from fusion.multimodal_fusion import MultimodalFusion, create_lexical_only_salsa, create_multimodal_salsa
from eval.evaluate import RecordingLevelEvaluator


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


class LOCOEvaluator:
    """Leave-One-Corpus-Out evaluator."""
    
    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = {}
        self.all_predictions = {}
        self.all_labels = {}
        self.all_group_ids = {}
    
    def _convert_manifest_to_csv(self, samples: List[Dict], csv_path: Path, 
                                corpus_configs: Dict[str, Dict], corpora: List[str]):
        """
        Convert JSON manifest format to CSV format for utterance dataset.
        This is a placeholder implementation - in practice, you'd need to 
        convert your specific manifest format to the expected CSV structure.
        """
        import pandas as pd
        
        # This is a placeholder - you'll need to implement based on your manifest structure
        rows = []
        for sample in samples:
            # Extract required fields based on your manifest structure
            row = {
                'recording_id': sample.get('recording_id', sample.get('id', 'unknown')),
                'utterance_id': sample.get('utterance_id', f"{sample.get('id', 'unknown')}_0"),
                'cognitive_decline': sample.get('label', 0),
                'corpus': sample.get('corpus', 'unknown'),
                'group_id': sample.get('group_id', 0)
            }
            
            # Add placeholder features if not present
            # You'll need to extract actual features from your data
            for i in range(384):  # MiniLM embeddings
                row[f'emb_{i}'] = 0.0  # Placeholder
            
            # Add handcrafted features
            handcrafted_features = [
                'utt_len', 'has_filler', 'start_time_s', 'end_time_s',
                'noun_count', 'verb_count', 'pronoun_count', 'adj_count',
                'noun_ratio', 'verb_ratio', 'pronoun_ratio', 'adj_ratio',
                'mean_dep_distance', 'subordinate_clause_count', 'total_dependencies',
                'semantic_coherence'
            ]
            
            for feature in handcrafted_features:
                row[feature] = 0.0  # Placeholder
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        self.logger.warning(f"Created placeholder CSV at {csv_path}. You need to implement proper feature extraction.")
    
    def run_loco_evaluation(self, 
                           corpus_configs: Dict[str, Dict],
                           experiment_name: str,
                           use_group_dro: bool = True) -> Dict:
        """
        Run complete LOCO evaluation.
        
        Args:
            corpus_configs: Dictionary mapping corpus names to their data configurations
            experiment_name: Name for this experiment
            use_group_dro: Whether to use GroupDRO training
            
        Returns:
            Dictionary containing all LOCO results
        """
        corpus_names = list(corpus_configs.keys())
        self.logger.info(f"Running LOCO evaluation on corpora: {corpus_names}")
        
        loco_results = {}
        
        # For each corpus as test set
        for test_corpus in corpus_names:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"LOCO Evaluation: Testing on {test_corpus}")
            self.logger.info(f"{'='*50}")
            
            # Get training corpora (all except test)
            train_corpora = [c for c in corpus_names if c != test_corpus]
            
            self.logger.info(f"Training on: {train_corpora}")
            self.logger.info(f"Testing on: {test_corpus}")
            
            # Train model on combined training corpora
            model = self._train_loco_model(
                train_corpora=train_corpora,
                test_corpus=test_corpus,
                corpus_configs=corpus_configs,
                experiment_name=f"{experiment_name}_loco_{test_corpus}",
                use_group_dro=use_group_dro
            )
            
            # Evaluate on test corpus
            test_metrics = self._evaluate_loco_model(
                model=model,
                test_corpus=test_corpus,
                corpus_configs=corpus_configs
            )
            
            loco_results[test_corpus] = test_metrics
            
            # Log results for this LOCO iteration
            self.logger.info(f"\nLOCO Results for {test_corpus}:")
            self.logger.info(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
            self.logger.info(f"  AUROC: {test_metrics['auroc']:.4f}")
            self.logger.info(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
            self.logger.info(f"  ECE: {test_metrics['ece']:.4f}")
            
            if 'worst_group_f1' in test_metrics:
                self.logger.info(f"  Worst Group F1: {test_metrics['worst_group_f1']:.4f}")
        
        # Compute aggregate LOCO statistics
        aggregate_stats = self._compute_aggregate_loco_stats(loco_results)
        
        # Save results
        results_dict = {
            'loco_results': loco_results,
            'aggregate_stats': aggregate_stats,
            'corpus_configs': corpus_configs,
            'config': self.config.__dict__,
            'use_group_dro': use_group_dro
        }
        
        return results_dict
    
    def _train_loco_model(self,
                         train_corpora: List[str],
                         test_corpus: str,
                         corpus_configs: Dict[str, Dict],
                         experiment_name: str,
                         use_group_dro: bool) -> torch.nn.Module:
        """Train model for LOCO evaluation."""
        self.logger.info(f"Training model for LOCO evaluation...")
        
        # Create combined training data
        combined_train_samples = []
        combined_val_samples = []
        
        for corpus_name in train_corpora:
            corpus_config = corpus_configs[corpus_name]
            
            # Load corpus data
            with open(corpus_config['train_manifest'], 'r') as f:
                train_samples = json.load(f)
            
            with open(corpus_config['val_manifest'], 'r') as f:
                val_samples = json.load(f)
            
            # Update group IDs to be corpus-specific
            group_id = corpus_config['group_id']
            for sample in train_samples:
                sample['group_id'] = group_id
                sample['corpus'] = corpus_name
            
            for sample in val_samples:
                sample['group_id'] = group_id
                sample['corpus'] = corpus_name
            
            combined_train_samples.extend(train_samples)
            combined_val_samples.extend(val_samples)
        
        # Create temporary CSV files from JSON manifests
        temp_dir = Path("./temp_loco")
        temp_dir.mkdir(exist_ok=True)
        
        train_csv_path = temp_dir / f"{experiment_name}_train.csv"
        val_csv_path = temp_dir / f"{experiment_name}_val.csv"
        
        # Convert JSON manifests to CSV format (assuming utterance-level data)d
        self._convert_manifest_to_csv(combined_train_samples, train_csv_path, corpus_configs, train_corpora)
        self._convert_manifest_to_csv(combined_val_samples, val_csv_path, corpus_configs, train_corpora)
        
        # Create data loaders using utterance dataset
        train_loader, val_loader, _ = create_data_loaders(
            train_csv=str(train_csv_path),
            val_csv=str(val_csv_path),
            test_csv=str(val_csv_path),  # Dummy test CSV
            batch_size=self.config.batch_size,
            num_workers=4,
            include_audio=True,
            include_lexical=True
        )
        
        # Create model
        model = create_multimodal_salsa(
            acoustic_dim=self.config.acoustic_dim,
            lexical_dim=self.config.lexical_dim,
            fusion_dim=self.config.hidden_dim,  # Use hidden_dim as fusion_dim
            num_classes=self.config.num_classes,
            aggregation_type='attention',
            dropout=self.config.dropout
        )
        
        # Setup training config for LOCO
        loco_config = TrainingConfig()
        loco_config.__dict__.update(self.config.__dict__)
        loco_config.use_group_dro = use_group_dro
        loco_config.max_epochs = min(self.config.max_epochs, 50)  # Limit epochs for LOCO
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=loco_config,
            device=self.device,
            experiment_name=experiment_name,
            log_dir="./logs/loco"
        )
        
        # Train model
        trainer.fit(train_loader, val_loader)
        
        # Clean up temporary files
        train_csv_path.unlink(missing_ok=True)
        val_csv_path.unlink(missing_ok=True)
        
        return trainer.model
    
    def _evaluate_loco_model(self,
                            model: torch.nn.Module,
                            test_corpus: str,
                            corpus_configs: Dict[str, Dict]) -> Dict:
        """Evaluate LOCO model on test corpus."""
        self.logger.info(f"Evaluating on {test_corpus}...")
        
        corpus_config = corpus_configs[test_corpus]
        
        # Create test data loader - assume CSV format
        test_csv = corpus_config.get('test_csv', corpus_config.get('test_manifest'))
        if not test_csv:
            raise ValueError(f"No test data specified for corpus {test_corpus}")
        
        # Create dummy train/val for the function signature
        _, _, test_loader = create_data_loaders(
            train_csv=test_csv,  # Dummy
            val_csv=test_csv,    # Dummy
            test_csv=test_csv,
            batch_size=self.config.batch_size,
            num_workers=4,
            include_audio=True,
            include_lexical=True
        )
        
        # Evaluate model
        model.eval()
        all_predictions = []
        all_labels = []
        all_group_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                lexical_features = batch['lexical_features'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get acoustic features if available
                acoustic_features = batch.get('acoustic_features')
                if acoustic_features is not None:
                    acoustic_features = acoustic_features.to(self.device)
                
                # Get group_ids if available, otherwise use corpus as group
                group_ids = batch.get('group_ids', torch.zeros_like(labels))
                if isinstance(group_ids, torch.Tensor):
                    group_ids = group_ids.to(self.device)
                
                outputs = model(
                    lexical_features=lexical_features,
                    acoustic_features=acoustic_features,
                    attention_mask=attention_mask
                )
                predictions = torch.softmax(outputs, dim=1)
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_group_ids.append(group_ids.cpu())
        
        # Concatenate results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_group_ids = torch.cat(all_group_ids, dim=0)
        
        # Compute metrics
        metrics = compute_classification_metrics(
            all_predictions,
            all_labels,
            all_group_ids,
            num_bootstrap=1000
        )
        
        # Store for later analysis
        self.all_predictions[test_corpus] = all_predictions
        self.all_labels[test_corpus] = all_labels
        self.all_group_ids[test_corpus] = all_group_ids
        
        return metrics
    
    def _compute_aggregate_loco_stats(self, loco_results: Dict) -> Dict:
        """Compute aggregate statistics across LOCO iterations."""
        metrics_to_aggregate = [
            'macro_f1', 'auroc', 'balanced_accuracy', 'ece', 'brier_score'
        ]
        
        aggregate_stats = {}
        
        for metric in metrics_to_aggregate:
            values = [results.get(metric, 0) for results in loco_results.values()]
            
            if values:
                aggregate_stats[f'{metric}_mean'] = np.mean(values)
                aggregate_stats[f'{metric}_std'] = np.std(values)
                aggregate_stats[f'{metric}_min'] = np.min(values)
                aggregate_stats[f'{metric}_max'] = np.max(values)
        
        # Compute worst-group metrics if available
        worst_group_f1_values = []
        for results in loco_results.values():
            if 'worst_group_f1' in results:
                worst_group_f1_values.append(results['worst_group_f1'])
        
        if worst_group_f1_values:
            aggregate_stats['worst_group_f1_mean'] = np.mean(worst_group_f1_values)
            aggregate_stats['worst_group_f1_std'] = np.std(worst_group_f1_values)
        
        return aggregate_stats
    
    def generate_loco_report(self, results: Dict, save_path: str):
        """Generate comprehensive LOCO evaluation report."""
        self.logger.info("Generating LOCO evaluation report...")
        
        # Create results dataframe
        loco_data = []
        for test_corpus, metrics in results['loco_results'].items():
            row = {'Test_Corpus': test_corpus}
            row.update(metrics)
            loco_data.append(row)
        
        df = pd.DataFrame(loco_data)
        
        # Save detailed results
        results_file = Path(save_path) / "loco_detailed_results.csv"
        df.to_csv(results_file, index=False)
        
        # Create summary report
        report_file = Path(save_path) / "loco_summary_report.txt"
        with open(report_file, 'w') as f:
            f.write("SALSA Leave-One-Corpus-Out (LOCO) Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Aggregate statistics
            f.write("Aggregate Statistics Across All LOCO Iterations:\n")
            f.write("-" * 50 + "\n")
            
            agg_stats = results['aggregate_stats']
            for metric in ['macro_f1', 'auroc', 'balanced_accuracy', 'ece']:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                if mean_key in agg_stats:
                    f.write(f"{metric.upper()}: {agg_stats[mean_key]:.4f} ± {agg_stats[std_key]:.4f}\n")
            
            f.write("\n")
            
            # Per-corpus results
            f.write("Per-Corpus Results:\n")
            f.write("-" * 30 + "\n")
            
            for test_corpus, metrics in results['loco_results'].items():
                f.write(f"\nTest Corpus: {test_corpus}\n")
                f.write(f"  Macro F1: {metrics.get('macro_f1', 0):.4f}\n")
                f.write(f"  AUROC: {metrics.get('auroc', 0):.4f}\n")
                f.write(f"  Balanced Acc: {metrics.get('balanced_accuracy', 0):.4f}\n")
                f.write(f"  ECE: {metrics.get('ece', 0):.4f}\n")
                if 'worst_group_f1' in metrics:
                    f.write(f"  Worst Group F1: {metrics['worst_group_f1']:.4f}\n")
        
        # Generate plots if we have stored predictions
        if self.all_predictions:
            viz = VisualizationUtils()
            
            # ROC curves
            roc_data = {corpus: (preds, labels) 
                       for corpus, (preds, labels) in 
                       zip(self.all_predictions.keys(), 
                           zip(self.all_predictions.values(), self.all_labels.values()))}
            
            viz.plot_roc_curves(roc_data, save_path=Path(save_path) / "loco_roc_curves.png")
            
            # PR curves
            viz.plot_precision_recall_curves(roc_data, save_path=Path(save_path) / "loco_pr_curves.png")
        
        self.logger.info(f"LOCO report saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='LOCO evaluation for SALSA')
    
    # Corpus configuration
    parser.add_argument('--corpus_config', type=str, required=True,
                       help='JSON file containing corpus configurations')
    
    # Model arguments
    parser.add_argument('--acoustic_dim', type=int, default=768)
    parser.add_argument('--lexical_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_fusion_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    
    # GroupDRO
    parser.add_argument('--use_group_dro', action='store_true',
                       help='Use GroupDRO training')
    parser.add_argument('--group_weights_step_size', type=float, default=0.01)
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./loco_results')
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
    
    logger.info(f"Starting LOCO evaluation: {args.experiment_name}")
    logger.info(f"Device: {device}")
    logger.info(f"GroupDRO: {args.use_group_dro}")
    
    # Load corpus configurations
    with open(args.corpus_config, 'r') as f:
        corpus_configs = json.load(f)
    
    logger.info(f"Loaded configurations for {len(corpus_configs)} corpora")
    for corpus_name in corpus_configs.keys():
        logger.info(f"  - {corpus_name}")
    
    # Create training config
    config = TrainingConfig()
    config.acoustic_dim = args.acoustic_dim
    config.lexical_dim = args.lexical_dim
    config.hidden_dim = args.hidden_dim
    config.num_classes = 2  # Binary classification
    config.num_fusion_layers = args.num_fusion_layers
    config.dropout = args.dropout
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.max_epochs = args.max_epochs
    config.patience = args.patience
    config.use_group_dro = args.use_group_dro
    config.group_weights_step_size = args.group_weights_step_size
    
    # Create LOCO evaluator
    evaluator = LOCOEvaluator(config, device)
    
    # Run LOCO evaluation
    results = evaluator.run_loco_evaluation(
        corpus_configs=corpus_configs,
        experiment_name=args.experiment_name,
        use_group_dro=args.use_group_dro
    )
    
    # Save results
    results_file = Path(args.output_dir) / f"{args.experiment_name}_loco_results.json"
    with open(results_file, 'w') as f:
        # Convert torch tensors to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key in ['loco_results', 'aggregate_stats', 'corpus_configs', 'config']:
                json_results[key] = value
            else:
                json_results[key] = str(value)
        
        json.dump(json_results, f, indent=2, default=str)
    
    # Generate comprehensive report
    evaluator.generate_loco_report(results, args.output_dir)
    
    # Final summary
    agg_stats = results['aggregate_stats']
    logger.info("\n" + "="*60)
    logger.info("LOCO EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Mean Macro F1: {agg_stats.get('macro_f1_mean', 0):.4f} ± {agg_stats.get('macro_f1_std', 0):.4f}")
    logger.info(f"Mean AUROC: {agg_stats.get('auroc_mean', 0):.4f} ± {agg_stats.get('auroc_std', 0):.4f}")
    logger.info(f"Mean ECE: {agg_stats.get('ece_mean', 0):.4f} ± {agg_stats.get('ece_std', 0):.4f}")
    
    if 'worst_group_f1_mean' in agg_stats:
        logger.info(f"Mean Worst Group F1: {agg_stats['worst_group_f1_mean']:.4f} ± {agg_stats['worst_group_f1_std']:.4f}")
    
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info("LOCO evaluation completed successfully!")


if __name__ == "__main__":
    main()