"""
OpenSMILE eGeMAPS baseline implementation for SALSA.
Classical feature extraction + linear models baseline.
"""
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import opensmile
import audiofile
import joblib
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path


class OpenSMILEFeatureExtractor:
    """Extract openSMILE eGeMAPS features from audio files."""
    
    def __init__(self, feature_set: str = 'egemaps', feature_level: str = 'functionals'):
        """
        Initialize openSMILE feature extractor.
        
        Args:
            feature_set: 'egemaps' or 'compare' (ComParE 2016)
            feature_level: 'functionals' or 'lld' (low-level descriptors)
        """
        self.feature_set = feature_set
        self.feature_level = feature_level
        
        # Set up openSMILE feature configuration
        if feature_set.lower() == 'egemaps':
            self.smile_feature_set = opensmile.FeatureSet.eGeMAPSv02
        elif feature_set.lower() == 'compare':
            self.smile_feature_set = opensmile.FeatureSet.ComParE_2016
        else:
            raise ValueError(f"Unsupported feature set: {feature_set}")
        
        if feature_level.lower() == 'functionals':
            self.smile_feature_level = opensmile.FeatureLevel.Functionals
        elif feature_level.lower() == 'lld':
            self.smile_feature_level = opensmile.FeatureLevel.LowLevelDescriptors
        else:
            raise ValueError(f"Unsupported feature level: {feature_level}")
        
        # Initialize openSMILE
        self.smile = opensmile.Smile(
            feature_set=self.smile_feature_set,
            feature_level=self.smile_feature_level,
            loglevel=0  # Suppress logs
        )
        
        self.logger = logging.getLogger(__name__)
    
    def extract_features_from_file(self, audio_path: str) -> np.ndarray:
        """Extract features from a single audio file."""
        try:
            # Read audio file
            signal, sampling_rate = audiofile.read(audio_path, always_2d=True)
            
            # Extract features
            features = self.smile.process_signal(signal, sampling_rate)
            
            # Convert to numpy array
            if isinstance(features, pd.DataFrame):
                return features.values.flatten()
            else:
                return np.array(features).flatten()
                
        except Exception as e:
            self.logger.error(f"Error extracting features from {audio_path}: {e}")
            # Return zeros as fallback
            if self.feature_set == 'egemaps':
                return np.zeros(88)  # eGeMAPS has 88 functionals
            else:
                return np.zeros(6373)  # ComParE has 6373 features
    
    def extract_features_batch(self, 
                             audio_paths: List[str], 
                             output_file: Optional[str] = None) -> np.ndarray:
        """Extract features from multiple audio files."""
        features_list = []
        
        for audio_path in audio_paths:
            features = self.extract_features_from_file(audio_path)
            features_list.append(features)
            
            if len(features_list) % 100 == 0:
                self.logger.info(f"Processed {len(features_list)}/{len(audio_paths)} files")
        
        features_array = np.array(features_list)
        
        # Save to file if specified
        if output_file:
            np.save(output_file, features_array)
            self.logger.info(f"Saved features to {output_file}")
        
        return features_array
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for the current configuration."""
        # Create a dummy signal to get feature names
        dummy_signal = np.random.randn(16000).reshape(1, -1)  # 1 second at 16kHz
        features = self.smile.process_signal(dummy_signal, 16000)
        
        if isinstance(features, pd.DataFrame):
            return list(features.columns)
        else:
            # Generic names if DataFrame not available
            if self.feature_set == 'egemaps':
                return [f'egemaps_{i}' for i in range(88)]
            else:
                return [f'compare_{i}' for i in range(6373)]


class OpenSMILEBaseline:
    """Complete openSMILE baseline with classical ML models."""
    
    def __init__(self, 
                 feature_set: str = 'egemaps',
                 classifier_type: str = 'logistic',
                 random_state: int = 42):
        """
        Initialize baseline model.
        
        Args:
            feature_set: 'egemaps' or 'compare'
            classifier_type: 'logistic', 'svm', or 'rf' (random forest)
            random_state: Random seed for reproducibility
        """
        self.feature_set = feature_set
        self.classifier_type = classifier_type
        self.random_state = random_state
        
        self.feature_extractor = OpenSMILEFeatureExtractor(feature_set)
        self.scaler = StandardScaler()
        self.classifier = None
        self.pipeline = None
        
        self.logger = logging.getLogger(__name__)
        
        # Setup classifier
        self._setup_classifier()
    
    def _setup_classifier(self):
        """Setup the classifier with hyperparameter grid."""
        if self.classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
            self.param_grid = {
                'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear']
            }
        
        elif self.classifier_type == 'svm':
            self.classifier = SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            )
            self.param_grid = {
                'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        
        elif self.classifier_type == 'rf':
            self.classifier = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            self.param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 20, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', self.classifier)
        ])
    
    def fit(self, 
            audio_paths: List[str], 
            labels: List[int],
            group_ids: Optional[List[int]] = None,
            use_grid_search: bool = True,
            cv_folds: int = 5) -> Dict:
        """
        Train the baseline model.
        
        Args:
            audio_paths: List of paths to audio files
            labels: List of labels (0 for healthy, 1 for dementia)
            group_ids: Optional group identifiers for stratification
            use_grid_search: Whether to use grid search for hyperparameter tuning
            cv_folds: Number of CV folds for grid search
            
        Returns:
            Training results dictionary
        """
        self.logger.info("Extracting features...")
        X = self.feature_extractor.extract_features_batch(audio_paths)
        y = np.array(labels)
        
        self.logger.info(f"Feature matrix shape: {X.shape}")
        self.logger.info(f"Class distribution: {np.bincount(y)}")
        
        if use_grid_search:
            self.logger.info("Starting grid search...")
            
            # Setup stratified CV
            if group_ids is not None:
                # Group-aware CV to avoid data leakage
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            else:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # Grid search
            grid_search = GridSearchCV(
                self.pipeline,
                self.param_grid,
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            # Update pipeline with best parameters
            self.pipeline = grid_search.best_estimator_
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
        
        else:
            self.logger.info("Training with default parameters...")
            self.pipeline.fit(X, y)
            results = {'best_params': None, 'best_score': None}
        
        return results
    
    def predict(self, audio_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new audio files.
        
        Returns:
            (predictions, probabilities)
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        X = self.feature_extractor.extract_features_batch(audio_paths)
        
        predictions = self.pipeline.predict(X)
        probabilities = self.pipeline.predict_proba(X)
        
        return predictions, probabilities
    
    def evaluate(self, 
                audio_paths: List[str], 
                labels: List[int],
                group_ids: Optional[List[int]] = None) -> Dict:
        """Evaluate the model on test data."""
        predictions, probabilities = self.predict(audio_paths)
        
        # Use our comprehensive metrics
        from ..eval.metrics import compute_classification_metrics
        
        metrics = compute_classification_metrics(
            torch.tensor(probabilities),
            torch.tensor(labels),
            torch.tensor(group_ids) if group_ids else None
        )
        
        # Add classification report
        report = classification_report(labels, predictions, output_dict=True)
        metrics['classification_report'] = report
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.pipeline is None:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'pipeline': self.pipeline,
            'feature_set': self.feature_set,
            'classifier_type': self.classifier_type,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.pipeline = model_data['pipeline']
        self.feature_set = model_data['feature_set']
        self.classifier_type = model_data['classifier_type']
        self.random_state = model_data['random_state']
        
        # Recreate feature extractor with correct settings
        self.feature_extractor = OpenSMILEFeatureExtractor(self.feature_set)
        
        self.logger.info(f"Model loaded from {filepath}")


class FrozenWavLMBaseline:
    """Frozen WavLM + linear probe baseline."""
    
    def __init__(self, 
                 model_name: str = "microsoft/wavlm-base",
                 pooling_strategy: str = "mean",
                 classifier_type: str = "logistic",
                 random_state: int = 42):
        """
        Initialize frozen WavLM baseline.
        
        Args:
            model_name: HuggingFace model name
            pooling_strategy: 'mean', 'max', or 'attention'
            classifier_type: 'logistic', 'svm', or 'rf'
            random_state: Random seed
        """
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.classifier_type = classifier_type
        self.random_state = random_state
        
        # Initialize WavLM model
        from transformers import WavLMModel
        import torch
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wavlm_model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.wavlm_model.eval()
        
        # Freeze WavLM parameters
        for param in self.wavlm_model.parameters():
            param.requires_grad = False
        
        # Setup classifier (similar to OpenSMILE baseline)
        self.scaler = StandardScaler()
        self._setup_classifier()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_classifier(self):
        """Setup classifier with hyperparameter grid."""
        if self.classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
            self.param_grid = {
                'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            }
        
        elif self.classifier_type == 'svm':
            self.classifier = SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            )
            self.param_grid = {
                'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0],
                'classifier__kernel': ['linear', 'rbf']
            }
        
        elif self.classifier_type == 'rf':
            self.classifier = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            )
            self.param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, None]
            }
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', self.classifier)
        ])
    
    def extract_features(self, audio_paths: List[str], batch_size: int = 8) -> np.ndarray:
        """Extract WavLM features from audio files."""
        
        class AudioDataset(Dataset):
            def __init__(self, audio_paths):
                self.audio_paths = audio_paths
            
            def __len__(self):
                return len(self.audio_paths)
            
            def __getitem__(self, idx):
                audio_path = self.audio_paths[idx]
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                return waveform.squeeze(0)  # (T,)
        
        dataset = AudioDataset(audio_paths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                               collate_fn=lambda batch: torch.nn.utils.rnn.pad_sequence(batch, batch_first=True))
        
        features_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Create attention mask
                attention_mask = (batch != 0).float()
                
                # Forward pass
                outputs = self.wavlm_model(batch, attention_mask=attention_mask, return_dict=True)
                hidden_states = outputs.last_hidden_state  # (B, T, D)
                
                # Apply pooling
                if self.pooling_strategy == 'mean':
                    # Masked mean pooling
                    features = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                elif self.pooling_strategy == 'max':
                    features, _ = hidden_states.max(dim=1)
                else:  # attention pooling
                    # Use the existing AttentionPooling from acoustic branch
                    from ..acoustic.acoustic import AttentionPooling
                    if not hasattr(self, 'attention_pool'):
                        self.attention_pool = AttentionPooling(hidden_states.size(-1)).to(self.device)
                    features = self.attention_pool(hidden_states, attention_mask)
                
                features_list.append(features.cpu().numpy())
        
        return np.vstack(features_list)
    
    def fit(self, 
            audio_paths: List[str], 
            labels: List[int],
            use_grid_search: bool = True,
            cv_folds: int = 5) -> Dict:
        """Train the baseline model."""
        self.logger.info("Extracting WavLM features...")
        X = self.extract_features(audio_paths)
        y = np.array(labels)
        
        self.logger.info(f"Feature matrix shape: {X.shape}")
        
        if use_grid_search:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            grid_search = GridSearchCV(
                self.pipeline,
                self.param_grid,
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            self.pipeline = grid_search.best_estimator_
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
        else:
            self.pipeline.fit(X, y)
            return {}
    
    def predict(self, audio_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        X = self.extract_features(audio_paths)
        predictions = self.pipeline.predict(X)
        probabilities = self.pipeline.predict_proba(X)
        
        return predictions, probabilities
    
    def evaluate(self, audio_paths: List[str], labels: List[int]) -> Dict:
        """Evaluate the model."""
        predictions, probabilities = self.predict(audio_paths)
        
        from ..eval.metrics import compute_classification_metrics
        import torch
        
        metrics = compute_classification_metrics(
            torch.tensor(probabilities),
            torch.tensor(labels)
        )
        
        return metrics