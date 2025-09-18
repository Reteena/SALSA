"""
Core training utilities for SALSA project including GroupDRO implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing for handling class imbalance.
    """
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.05, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        num_classes = inputs.size(-1)
        smoothed_targets = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GroupDROLoss(nn.Module):
    """
    Group Distributionally Robust Optimization (GroupDRO) loss.
    Optimizes for worst-group performance to improve generalization.
    """
    def __init__(self, 
                 num_groups: int,
                 group_weights_step_size: float = 0.01,
                 base_loss_fn: Optional[nn.Module] = None,
                 is_robust: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.group_weights_step_size = group_weights_step_size
        self.is_robust = is_robust
        
        # Initialize uniform group weights
        self.register_buffer('group_weights', torch.ones(num_groups) / num_groups)
        self.register_buffer('group_counts', torch.zeros(num_groups))
        
        # Base loss function
        self.base_loss_fn = base_loss_fn if base_loss_fn else nn.CrossEntropyLoss(reduction='none')
        
        # For tracking group losses
        self.group_losses_ema = torch.zeros(num_groups)
    
    def forward(self, outputs, targets, group_ids):
        """
        Args:
            outputs: Model predictions (B, num_classes)
            targets: True labels (B,)
            group_ids: Group identifiers (B,)
        """
        device = outputs.device
        batch_size = outputs.size(0)
        
        # Move group weights to device if needed
        if self.group_weights.device != device:
            self.group_weights = self.group_weights.to(device)
            self.group_counts = self.group_counts.to(device)
            self.group_losses_ema = self.group_losses_ema.to(device)
        
        # Compute per-sample losses
        losses = self.base_loss_fn(outputs, targets)
        
        # Compute group losses
        group_losses = torch.zeros(self.num_groups, device=device)
        group_counts = torch.zeros(self.num_groups, device=device)
        
        for g in range(self.num_groups):
            group_mask = (group_ids == g)
            if group_mask.sum() > 0:
                group_losses[g] = losses[group_mask].mean()
                group_counts[g] = group_mask.sum().float()
        
        # Update group counts
        self.group_counts += group_counts
        
        if self.is_robust:
            # GroupDRO: weighted combination of group losses
            total_loss = (self.group_weights * group_losses).sum()
            
            # Update group weights (moving towards worst-performing groups)
            with torch.no_grad():
                # Update EMA of group losses
                alpha = 0.1  # EMA decay
                self.group_losses_ema = alpha * group_losses + (1 - alpha) * self.group_losses_ema
                
                # Update group weights
                self.group_weights = self.group_weights * torch.exp(
                    self.group_weights_step_size * self.group_losses_ema
                )
                self.group_weights = self.group_weights / self.group_weights.sum()
        else:
            # ERM: simple average
            total_loss = losses.mean()
        
        return total_loss, group_losses, group_counts
    
    def get_group_info(self):
        """Get current group weights and losses for logging."""
        return {
            'group_weights': self.group_weights.cpu().numpy(),
            'group_losses_ema': self.group_losses_ema.cpu().numpy(),
            'group_counts': self.group_counts.cpu().numpy()
        }


class TrainingConfig:
    """Configuration for training experiments."""
    def __init__(self):
        # Model hyperparameters
        self.acoustic_dim = 768
        self.lexical_dim = 512
        self.hidden_dim = 768
        self.num_classes = 2
        self.num_fusion_layers = 2
        self.dropout = 0.3
        
        # LoRA hyperparameters
        self.lora_rank = 8
        self.lora_alpha = 16
        
        # Training hyperparameters
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.max_epochs = 100
        self.patience = 15
        self.gradient_clip = 1.0
        
        # Loss function parameters
        self.focal_alpha = 1.0
        self.focal_gamma = 2.0
        self.label_smoothing = 0.05
        
        # GroupDRO parameters
        self.use_group_dro = True
        self.group_weights_step_size = 0.01
        
        # Scheduler parameters
        self.scheduler_type = 'plateau'  # 'plateau' or 'cosine'
        self.scheduler_patience = 5
        self.scheduler_factor = 0.5
        
        # Logging and checkpointing
        self.log_interval = 10
        self.save_best_only = True
        self.use_wandb = False
        
        # Evaluation
        self.eval_interval = 1
        self.bootstrap_samples = 1000


class Trainer:
    """
    Main trainer class for SALSA models with support for ERM and GroupDRO.
    """
    def __init__(self, 
                 model: nn.Module,
                 config: TrainingConfig,
                 device: torch.device,
                 experiment_name: str = "salsa_experiment",
                 log_dir: str = "./logs"):
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.experiment_name = experiment_name
        
        # Setup logging
        self.logger = self._setup_logging(log_dir)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        base_loss = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing
        )
        
        # Will be set based on number of groups in data
        self.loss_fn = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = -float('inf')
        self.patience_counter = 0
        self.training_history = defaultdict(list)
    
    def _setup_logging(self, log_dir):
        """Setup logging configuration."""
        import os
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(f"{log_dir}/{self.experiment_name}.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for different components."""
        # Separate parameters for different components
        lora_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                lora_params.append(param)
            else:
                other_params.append(param)
        
        # Different learning rates for LoRA and other parameters
        optimizer_params = [
            {'params': lora_params, 'lr': self.config.learning_rate * 2},  # Higher LR for LoRA
            {'params': other_params, 'lr': self.config.learning_rate}
        ]
        
        return AdamW(optimizer_params, weight_decay=self.config.weight_decay)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                verbose=True
            )
        elif self.config.scheduler_type == 'cosine':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                verbose=True
            )
        else:
            return None
    
    def setup_loss_function(self, num_groups: int):
        """Setup loss function after knowing the number of groups."""
        base_loss = FocalLoss(
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
            label_smoothing=self.config.label_smoothing
        )
        
        self.loss_fn = GroupDROLoss(
            num_groups=num_groups,
            group_weights_step_size=self.config.group_weights_step_size,
            base_loss_fn=base_loss,
            is_robust=self.config.use_group_dro
        ).to(self.device)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        group_losses_sum = defaultdict(float)
        group_counts_sum = defaultdict(int)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            acoustic_features = batch['acoustic_features'].to(self.device)
            lexical_features = batch['lexical_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            group_ids = batch['group_ids'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(acoustic_features, lexical_features)
            
            # Compute loss
            loss, group_losses, group_counts = self.loss_fn(outputs, labels, group_ids)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * acoustic_features.size(0)
            total_samples += acoustic_features.size(0)
            
            # Accumulate group statistics
            for g in range(len(group_losses)):
                if group_counts[g] > 0:
                    group_losses_sum[g] += group_losses[g].item() * group_counts[g].item()
                    group_counts_sum[g] += group_counts[g].item()
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}: '
                    f'Loss: {loss.item():.4f}'
                )
        
        # Compute average losses
        avg_loss = total_loss / total_samples
        group_avg_losses = {g: group_losses_sum[g] / group_counts_sum[g] 
                           for g in group_losses_sum.keys() if group_counts_sum[g] > 0}
        
        metrics = {
            'train_loss': avg_loss,
            'group_losses': group_avg_losses,
            **self.loss_fn.get_group_info()
        }
        
        return metrics
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        all_group_ids = []
        
        with torch.no_grad():
            for batch in val_loader:
                acoustic_features = batch['acoustic_features'].to(self.device)
                lexical_features = batch['lexical_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                group_ids = batch['group_ids'].to(self.device)
                
                outputs = self.model(acoustic_features, lexical_features)
                loss, _, _ = self.loss_fn(outputs, labels, group_ids)
                
                total_loss += loss.item() * acoustic_features.size(0)
                total_samples += acoustic_features.size(0)
                
                # Store predictions for metrics calculation
                predictions = torch.softmax(outputs, dim=1)
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_group_ids.append(group_ids.cpu())
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_group_ids = torch.cat(all_group_ids, dim=0)
        
        avg_loss = total_loss / total_samples
        
        return {
            'val_loss': avg_loss,
            'predictions': all_predictions,
            'labels': all_labels,
            'group_ids': all_group_ids
        }
    
    def fit(self, train_loader, val_loader):
        """Main training loop."""
        # Setup loss function with correct number of groups
        num_groups = len(set(train_loader.dataset.group_ids))
        self.setup_loss_function(num_groups)
        
        self.logger.info(f"Starting training with {num_groups} groups")
        self.logger.info(f"GroupDRO enabled: {self.config.use_group_dro}")
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate(val_loader)
                
                # Compute validation metrics using external metrics module
                try:
                    from eval.metrics import compute_classification_metrics
                    val_scores = compute_classification_metrics(
                        val_metrics['predictions'], 
                        val_metrics['labels'],
                        val_metrics['group_ids']
                    )
                except ImportError:
                    # Fallback to basic metrics if eval.metrics not available
                    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                    y_true = val_metrics['labels'].numpy()
                    y_pred = val_metrics['predictions'].argmax(dim=1).numpy()
                    y_scores = val_metrics['predictions'][:, 1].numpy()
                    
                    val_scores = {
                        'accuracy': accuracy_score(y_true, y_pred),
                        'macro_f1': f1_score(y_true, y_pred, average='macro'),
                        'auroc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
                    }
                
                # Log metrics
                self.logger.info(f"Epoch {epoch}:")
                self.logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
                self.logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                self.logger.info(f"  Val Macro F1: {val_scores['macro_f1']:.4f}")
                self.logger.info(f"  Val AUROC: {val_scores['auroc']:.4f}")
                
                # Update scheduler
                if self.scheduler:
                    if self.config.scheduler_type == 'plateau':
                        self.scheduler.step(val_scores['macro_f1'])
                    else:
                        self.scheduler.step()
                
                # Early stopping check
                current_score = val_scores['macro_f1']
                if current_score > self.best_val_score:
                    self.best_val_score = current_score
                    self.patience_counter = 0
                    
                    # Save best model
                    if self.config.save_best_only:
                        self._save_checkpoint(epoch, val_scores)
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.config.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.logger.info("Training completed")
        return self.training_history
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'best_val_score': self.best_val_score
        }
        
        checkpoint_path = f"./checkpoints/{self.experiment_name}_best.pt"
        import os
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if hasattr(self, 'loss_fn') and self.loss_fn:
            self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_score = checkpoint['best_val_score']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['metrics']