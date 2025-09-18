"""
Comprehensive evaluation script for SALSA with utterance-level features.
Handles evaluation of models trained on utterance-level data with recording-level aggregation.
"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.utterance_dataset import UtteranceDataset, collate_utterance_batch
from fusion.multimodal_fusion import MultimodalFusion
from eval.metrics import MetricsCalculator, VisualizationUtils, compute_classification_metrics
from train.trainer import TrainingConfig


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


class RecordingLevelEvaluator:
    """Evaluator for models that process utterance sequences and predict at recording level."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: torch.device,
                 config: Optional[TrainingConfig] = None):
        self.model = model.to(device)
        self.device = device
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger(__name__)
        
        self.model.eval()
    
    def evaluate_dataset(self, 
                        dataset: UtteranceDataset,
                        batch_size: int = 32,
                        detailed_results: bool = False) -> Dict:
        """
        Evaluate model on a dataset with utterance-level features.
        
        Args:
            dataset: UtteranceDataset instance
            batch_size: Batch size for evaluation
            detailed_results: Whether to return detailed per-recording results
            
        Returns:
            Dictionary containing metrics and optionally detailed results
        """
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_utterance_batch,
            num_workers=4,
            pin_memory=True
        )
        
        all_predictions = []
        all_labels = []
        all_recording_ids = []
        detailed_records = []
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                features = batch['features'].to(self.device)  # [B, max_seq_len, feature_dim]
                attention_mask = batch['attention_mask'].to(self.device)  # [B, max_seq_len]
                labels = batch['labels'].to(self.device)  # [B]
                recording_ids = batch['recording_ids']  # List of recording IDs
                
                # Forward pass
                outputs = self.model(features, attention_mask=attention_mask)
                predictions = outputs['predictions']  # [B, num_classes]
                
                # Convert to probabilities if needed
                if predictions.shape[1] == 2:
                    probs = torch.softmax(predictions, dim=1)
                else:
                    probs = torch.sigmoid(predictions)
                
                # Store results
                all_predictions.append(probs.cpu())
                all_labels.append(labels.cpu())
                all_recording_ids.extend(recording_ids)
                
                total_samples += len(recording_ids)
                
                # Store detailed results if requested
                if detailed_results:
                    for i, rec_id in enumerate(recording_ids):
                        record = {
                            'recording_id': rec_id,
                            'true_label': int(labels[i].cpu()),
                            'predicted_probs': probs[i].cpu().numpy().tolist(),
                            'predicted_label': int(torch.argmax(probs[i]).cpu()),
                            'num_utterances': int(torch.sum(attention_mask[i]).cpu()),
                            'confidence': float(torch.max(probs[i]).cpu())
                        }
                        detailed_records.append(record)
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Evaluated {total_samples} recordings...")
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute comprehensive metrics
        metrics = compute_classification_metrics(
            all_predictions, 
            all_labels,
            num_bootstrap=1000
        )
        
        # Add additional analysis
        metrics['total_recordings'] = len(all_recording_ids)
        metrics['unique_recordings'] = len(set(all_recording_ids))
        
        # Confidence analysis
        if all_predictions.shape[1] == 2:
            max_probs = torch.max(all_predictions, dim=1)[0]
        else:
            max_probs = all_predictions.squeeze()
        
        metrics['mean_confidence'] = float(torch.mean(max_probs))
        metrics['std_confidence'] = float(torch.std(max_probs))
        
        # Class distribution
        unique_labels, counts = torch.unique(all_labels, return_counts=True)
        class_distribution = {f'class_{int(label)}': int(count) 
                           for label, count in zip(unique_labels, counts)}
        metrics['class_distribution'] = class_distribution
        
        result = {
            'metrics': metrics,
            'predictions': all_predictions.numpy(),
            'labels': all_labels.numpy(),
            'recording_ids': all_recording_ids
        }
        
        if detailed_results:
            result['detailed_records'] = detailed_records
        
        return result
    
    def compare_models(self, 
                      models: Dict[str, torch.nn.Module],
                      dataset: UtteranceDataset,
                      batch_size: int = 32,
                      save_plots: bool = True,
                      output_dir: str = "./eval_results") -> Dict:
        """
        Compare multiple models on the same dataset.
        
        Args:
            models: Dictionary of model_name -> model
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            save_plots: Whether to save comparison plots
            output_dir: Directory to save results
            
        Returns:
            Dictionary with comparison results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        predictions_dict = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Temporarily switch models
            original_model = self.model
            self.model = model.to(self.device)
            
            # Evaluate
            result = self.evaluate_dataset(dataset, batch_size=batch_size)
            all_results[model_name] = result
            
            # Store for visualization
            predictions_dict[model_name] = (
                result['predictions'], 
                result['labels']
            )
            
            # Restore original model
            self.model = original_model
        
        # Create comparison visualizations
        if save_plots:
            viz_utils = VisualizationUtils()
            
            # ROC curves
            viz_utils.plot_roc_curves(
                predictions_dict, 
                save_path=f"{output_dir}/roc_comparison.png"
            )
            
            # PR curves
            viz_utils.plot_precision_recall_curves(
                predictions_dict,
                save_path=f"{output_dir}/pr_comparison.png"
            )
        
        # Create comparison table
        comparison_df = self._create_comparison_table(all_results)
        comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
        
        # Save detailed results
        with open(f"{output_dir}/detailed_results.json", 'w') as f:
            json.dump({
                name: {
                    'metrics': result['metrics'],
                    'recording_ids': result['recording_ids']
                }
                for name, result in all_results.items()
            }, f, indent=2, default=str)
        
        return {
            'results': all_results,
            'comparison_table': comparison_df,
            'output_dir': output_dir
        }
    
    def _create_comparison_table(self, results: Dict) -> pd.DataFrame:
        """Create comparison table from evaluation results."""
        rows = []
        
        for model_name, result in results.items():
            metrics = result['metrics']
            
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Balanced_Acc': f"{metrics['balanced_accuracy']:.3f}",
                'Macro_F1': f"{metrics['macro_f1']:.3f}",
                'AUROC': f"{metrics['auroc']:.3f}",
                'AUPRC': f"{metrics['auprc']:.3f}",
                'ECE': f"{metrics['ece']:.3f}",
                'Brier_Score': f"{metrics['brier_score']:.3f}",
                'F1_Healthy': f"{metrics['f1_healthy']:.3f}",
                'F1_Dementia': f"{metrics['f1_dementia']:.3f}",
                'Mean_Confidence': f"{metrics['mean_confidence']:.3f}",
                'Num_Recordings': metrics['total_recordings']
            }
            
            # Add confidence intervals if available
            if 'macro_f1_ci' in metrics:
                ci_lower, ci_upper = metrics['macro_f1_ci']
                row['F1_CI'] = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            
            if 'auroc_ci' in metrics:
                ci_lower, ci_upper = metrics['auroc_ci']
                row['AUROC_CI'] = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            
            rows.append(row)
        
        return pd.DataFrame(rows)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate SALSA models")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--data_csv", type=str, required=True,
                       help="Path to evaluation CSV file")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--detailed", action="store_true",
                       help="Save detailed per-recording results")
    parser.add_argument("--config_path", type=str,
                       help="Path to training config JSON")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Load config
    config = TrainingConfig()
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_csv}")
    dataset = UtteranceDataset(args.data_csv)
    logger.info(f"Loaded {len(dataset)} recordings with {dataset.get_total_utterances()} total utterances")
    
    # Create model
    model = MultimodalFusion(
        lexical_dim=config.lexical_dim,
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
        num_fusion_layers=config.num_fusion_layers,
        dropout=config.dropout,
        aggregation_strategy=config.aggregation_strategy
    )
    
    # Load checkpoint
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create evaluator
    evaluator = RecordingLevelEvaluator(model, device, config)
    
    # Evaluate
    logger.info("Starting evaluation...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    result = evaluator.evaluate_dataset(
        dataset, 
        batch_size=args.batch_size,
        detailed_results=args.detailed
    )
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    metrics = result['metrics']
    logger.info(f"Total Recordings: {metrics['total_recordings']}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"AUROC: {metrics['auroc']:.4f}")
    logger.info(f"AUPRC: {metrics['auprc']:.4f}")
    logger.info(f"ECE: {metrics['ece']:.4f}")
    logger.info(f"Brier Score: {metrics['brier_score']:.4f}")
    logger.info(f"F1 Healthy: {metrics['f1_healthy']:.4f}")
    logger.info(f"F1 Dementia: {metrics['f1_dementia']:.4f}")
    
    if 'macro_f1_ci' in metrics:
        ci_lower, ci_upper = metrics['macro_f1_ci']
        logger.info(f"Macro F1 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    if 'auroc_ci' in metrics:
        ci_lower, ci_upper = metrics['auroc_ci']
        logger.info(f"AUROC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    logger.info(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
    logger.info(f"Std Confidence: {metrics['std_confidence']:.4f}")
    
    # Save results
    results_path = f"{args.output_dir}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'config': config.__dict__,
            'args': vars(args)
        }, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_path}")
    
    # Create visualizations
    viz_utils = VisualizationUtils()
    
    # Confusion matrix
    viz_utils.plot_confusion_matrix(
        result['predictions'],
        result['labels'],
        save_path=f"{args.output_dir}/confusion_matrix.png"
    )
    
    # Calibration diagram
    viz_utils.plot_calibration_diagram(
        result['predictions'],
        result['labels'],
        save_path=f"{args.output_dir}/calibration_diagram.png"
    )
    
    # Save detailed results if requested
    if args.detailed and 'detailed_records' in result:
        detailed_df = pd.DataFrame(result['detailed_records'])
        detailed_df.to_csv(f"{args.output_dir}/detailed_results.csv", index=False)
        logger.info(f"Detailed results saved to {args.output_dir}/detailed_results.csv")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()