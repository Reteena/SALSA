"""
Comprehensive evaluation metrics for SALSA project.
Includes classification metrics, calibration, fairness analysis, and visualization.
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
try:
    from sklearn.metrics import calibration_curve
except ImportError:
    from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Comprehensive metrics calculator for AD detection."""
    
    def __init__(self, num_bootstrap_samples: int = 1000):
        self.num_bootstrap_samples = num_bootstrap_samples
    
    def compute_classification_metrics(self, 
                                     predictions: torch.Tensor,
                                     labels: torch.Tensor,
                                     group_ids: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            predictions: Model predictions (B, num_classes) or (B,) for probabilities
            labels: True labels (B,)
            group_ids: Optional group identifiers for group-wise metrics (B,)
            
        Returns:
            Dictionary of computed metrics
        """
        # Convert to numpy
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        if group_ids is not None and torch.is_tensor(group_ids):
            group_ids = group_ids.detach().cpu().numpy()
        
        # Handle predictions format
        if predictions.ndim == 2:
            # Softmax outputs - take positive class probability
            probs = predictions[:, 1] if predictions.shape[1] == 2 else predictions
            preds = (probs >= 0.5).astype(int)
        else:
            # Already probabilities
            probs = predictions
            preds = (probs >= 0.5).astype(int)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['balanced_accuracy'] = balanced_accuracy_score(labels, preds)
        metrics['macro_f1'] = f1_score(labels, preds, average='macro')
        metrics['weighted_f1'] = f1_score(labels, preds, average='weighted')
        metrics['precision'] = precision_score(labels, preds, average='macro')
        metrics['recall'] = recall_score(labels, preds, average='macro')
        
        # ROC and PR metrics
        try:
            metrics['auroc'] = roc_auc_score(labels, probs)
            metrics['auprc'] = average_precision_score(labels, probs)
        except ValueError:
            # Handle case where only one class is present
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
        
        # Class-specific metrics
        f1_scores = f1_score(labels, preds, average=None)
        metrics['f1_healthy'] = f1_scores[0] if len(f1_scores) > 0 else 0.0
        metrics['f1_dementia'] = f1_scores[1] if len(f1_scores) > 1 else 0.0
        
        # Calibration metrics
        calibration_metrics = self.compute_calibration_metrics(probs, labels)
        metrics.update(calibration_metrics)
        
        # Group-wise metrics if group_ids provided
        if group_ids is not None:
            group_metrics = self.compute_group_metrics(probs, labels, group_ids)
            metrics.update(group_metrics)
        
        return metrics
    
    def compute_calibration_metrics(self, probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute calibration metrics (ECE, Brier score)."""
        metrics = {}
        
        # Expected Calibration Error (ECE)
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                labels, probs, n_bins=10, strategy='quantile'
            )
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            metrics['ece'] = ece
        except:
            metrics['ece'] = 0.0
        
        # Brier Score
        brier_score = np.mean((probs - labels) ** 2)
        metrics['brier_score'] = brier_score
        
        # Maximum Calibration Error (MCE)
        try:
            mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
            metrics['mce'] = mce
        except:
            metrics['mce'] = 0.0
        
        return metrics
    
    def compute_group_metrics(self, 
                            probs: np.ndarray, 
                            labels: np.ndarray, 
                            group_ids: np.ndarray) -> Dict[str, float]:
        """Compute group-wise metrics for fairness analysis."""
        unique_groups = np.unique(group_ids)
        group_metrics = {}
        
        group_f1_scores = []
        group_aurocs = []
        
        for group in unique_groups:
            group_mask = (group_ids == group)
            if np.sum(group_mask) == 0:
                continue
            
            group_probs = probs[group_mask]
            group_labels = labels[group_mask]
            
            if len(np.unique(group_labels)) < 2:
                # Skip if only one class in group
                continue
            
            # Compute group-specific metrics
            group_preds = (group_probs >= 0.5).astype(int)
            
            try:
                group_f1 = f1_score(group_labels, group_preds, average='macro')
                group_auroc = roc_auc_score(group_labels, group_probs)
                
                group_f1_scores.append(group_f1)
                group_aurocs.append(group_auroc)
                
                group_metrics[f'f1_group_{int(group)}'] = group_f1
                group_metrics[f'auroc_group_{int(group)}'] = group_auroc
                group_metrics[f'count_group_{int(group)}'] = np.sum(group_mask)
                
            except ValueError:
                continue
        
        # Compute worst-group metrics
        if group_f1_scores:
            group_metrics['worst_group_f1'] = min(group_f1_scores)
            group_metrics['best_group_f1'] = max(group_f1_scores)
            group_metrics['f1_group_gap'] = max(group_f1_scores) - min(group_f1_scores)
        
        if group_aurocs:
            group_metrics['worst_group_auroc'] = min(group_aurocs)
            group_metrics['best_group_auroc'] = max(group_aurocs)
            group_metrics['auroc_group_gap'] = max(group_aurocs) - min(group_aurocs)
        
        return group_metrics
    
    def bootstrap_confidence_intervals(self, 
                                     predictions: np.ndarray,
                                     labels: np.ndarray,
                                     metric_name: str = 'macro_f1',
                                     confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence intervals for a specific metric.
        
        Returns:
            (metric_value, lower_bound, upper_bound)
        """
        n_samples = len(labels)
        bootstrap_scores = []
        
        for _ in range(self.num_bootstrap_samples):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_preds = predictions[indices]
            boot_labels = labels[indices]
            
            # Compute metric for bootstrap sample
            if boot_preds.ndim == 2:
                boot_probs = boot_preds[:, 1] if boot_preds.shape[1] == 2 else boot_preds
            else:
                boot_probs = boot_preds
            
            boot_preds_binary = (boot_probs >= 0.5).astype(int)
            
            if metric_name == 'macro_f1':
                score = f1_score(boot_labels, boot_preds_binary, average='macro')
            elif metric_name == 'auroc':
                try:
                    score = roc_auc_score(boot_labels, boot_probs)
                except ValueError:
                    continue
            elif metric_name == 'accuracy':
                score = accuracy_score(boot_labels, boot_preds_binary)
            elif metric_name == 'balanced_accuracy':
                score = balanced_accuracy_score(boot_labels, boot_preds_binary)
            else:
                continue
            
            bootstrap_scores.append(score)
        
        if not bootstrap_scores:
            return 0.0, 0.0, 0.0
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        metric_value = np.mean(bootstrap_scores)
        lower_bound = np.percentile(bootstrap_scores, lower_percentile)
        upper_bound = np.percentile(bootstrap_scores, upper_percentile)
        
        return metric_value, lower_bound, upper_bound
    
    def demographic_parity_difference(self, 
                                    predictions: np.ndarray,
                                    group_ids: np.ndarray) -> float:
        """Compute demographic parity difference between groups."""
        unique_groups = np.unique(group_ids)
        if len(unique_groups) < 2:
            return 0.0
        
        positive_rates = []
        for group in unique_groups:
            group_mask = (group_ids == group)
            if np.sum(group_mask) == 0:
                continue
            
            group_preds = predictions[group_mask]
            if predictions.ndim == 2:
                group_preds = (group_preds[:, 1] >= 0.5).astype(int)
            else:
                group_preds = (group_preds >= 0.5).astype(int)
            
            positive_rate = np.mean(group_preds)
            positive_rates.append(positive_rate)
        
        if len(positive_rates) < 2:
            return 0.0
        
        return max(positive_rates) - min(positive_rates)
    
    def equalized_odds_difference(self, 
                                predictions: np.ndarray,
                                labels: np.ndarray,
                                group_ids: np.ndarray) -> Dict[str, float]:
        """Compute equalized odds metrics."""
        unique_groups = np.unique(group_ids)
        if len(unique_groups) < 2:
            return {'tpr_diff': 0.0, 'fpr_diff': 0.0}
        
        tprs, fprs = [], []
        
        for group in unique_groups:
            group_mask = (group_ids == group)
            if np.sum(group_mask) == 0:
                continue
            
            group_preds = predictions[group_mask]
            group_labels = labels[group_mask]
            
            if predictions.ndim == 2:
                group_preds = (group_preds[:, 1] >= 0.5).astype(int)
            else:
                group_preds = (group_preds >= 0.5).astype(int)
            
            # Compute TPR and FPR for this group
            tn, fp, fn, tp = confusion_matrix(group_labels, group_preds, labels=[0, 1]).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tprs.append(tpr)
            fprs.append(fpr)
        
        return {
            'tpr_diff': max(tprs) - min(tprs) if tprs else 0.0,
            'fpr_diff': max(fprs) - min(fprs) if fprs else 0.0
        }


class VisualizationUtils:
    """Utilities for creating evaluation plots and visualizations."""
    
    @staticmethod
    def plot_roc_curves(predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                       save_path: Optional[str] = None):
        """Plot ROC curves for multiple models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, (probs, labels) in predictions_dict.items():
            if probs.ndim == 2:
                probs = probs[:, 1]
            
            try:
                fpr, tpr, _ = roc_curve(labels, probs)
                auc = roc_auc_score(labels, probs)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
            except ValueError:
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curves(predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                   save_path: Optional[str] = None):
        """Plot precision-recall curves for multiple models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, (probs, labels) in predictions_dict.items():
            if probs.ndim == 2:
                probs = probs[:, 1]
            
            try:
                precision, recall, _ = precision_recall_curve(labels, probs)
                ap = average_precision_score(labels, probs)
                plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
            except ValueError:
                continue
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_calibration_diagram(predictions: np.ndarray, 
                               labels: np.ndarray,
                               model_name: str = "Model",
                               save_path: Optional[str] = None):
        """Plot reliability diagram for calibration analysis."""
        if predictions.ndim == 2:
            predictions = predictions[:, 1]
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, predictions, n_bins=10, strategy='quantile'
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
        plt.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Perfect calibration")
        
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Reliability Diagram")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(predictions: np.ndarray, 
                            labels: np.ndarray,
                            class_names: List[str] = ["Healthy", "Dementia"],
                            save_path: Optional[str] = None):
        """Plot confusion matrix."""
        if predictions.ndim == 2:
            preds = (predictions[:, 1] >= 0.5).astype(int)
        else:
            preds = (predictions >= 0.5).astype(int)
        
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_group_performance(group_metrics: Dict[str, float],
                             metric_name: str = 'f1',
                             save_path: Optional[str] = None):
        """Plot performance across different groups."""
        # Extract group metrics
        groups = []
        scores = []
        
        for key, value in group_metrics.items():
            if key.startswith(f'{metric_name}_group_'):
                group_id = key.split('_')[-1]
                groups.append(f"Group {group_id}")
                scores.append(value)
        
        if not groups:
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(groups, scores, color=['skyblue' if score == max(scores) else 'lightcoral' for score in scores])
        plt.xlabel('Groups')
        plt.ylabel(f'{metric_name.upper()} Score')
        plt.title(f'{metric_name.upper()} Performance Across Groups')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Convenience function for main metrics computation
def compute_classification_metrics(predictions: torch.Tensor,
                                 labels: torch.Tensor,
                                 group_ids: Optional[torch.Tensor] = None,
                                 num_bootstrap: int = 1000) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics with confidence intervals.
    
    This is the main function used by the trainer.
    """
    calculator = MetricsCalculator(num_bootstrap_samples=num_bootstrap)
    metrics = calculator.compute_classification_metrics(predictions, labels, group_ids)
    
    # Add bootstrap confidence intervals for key metrics
    if isinstance(predictions, torch.Tensor):
        preds_np = predictions.detach().cpu().numpy()
    else:
        preds_np = predictions
        
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels
    
    # Bootstrap CIs for macro F1
    f1_mean, f1_lower, f1_upper = calculator.bootstrap_confidence_intervals(
        preds_np, labels_np, 'macro_f1'
    )
    metrics['macro_f1_ci'] = (f1_lower, f1_upper)
    
    # Bootstrap CIs for AUROC
    auroc_mean, auroc_lower, auroc_upper = calculator.bootstrap_confidence_intervals(
        preds_np, labels_np, 'auroc'
    )
    metrics['auroc_ci'] = (auroc_lower, auroc_upper)
    
    return metrics