"""
Dataset classes for utterance-level features with recording-level aggregation.
Handles loading CSV files with utterance features and grouping by recording_id.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Feature constants - hardcoded to match lexical_features.py output
EMB_DIM = 384
EMB_NAMES = [f"emb_{i}" for i in range(EMB_DIM)]
HANDCRAFTED_KEYS = [
    "utt_len", "has_filler", "start_time_s", "end_time_s",
    "noun_count", "verb_count", "pronoun_count", "adj_count",
    "noun_ratio", "verb_ratio", "pronoun_ratio", "adj_ratio",
    "mean_dep_distance", "subordinate_clause_count", "total_dependencies"
]
ALL_FEATURE_NAMES = EMB_NAMES + HANDCRAFTED_KEYS + ["semantic_coherence"]


class UtteranceDataset(Dataset):
    """
    Dataset for utterance-level features with recording-level labels.
    
    Loads CSV files produced by lexical_features.py with columns:
    recording_id, utterance_id, cognitive_decline, emb_0, ..., emb_383, 
    utt_len, ..., semantic_coherence
    
    Returns recording-level samples with variable number of utterances.
    """
    
    def __init__(
        self, 
        csv_path: str,
        audio_dir: Optional[str] = None,
        sample_rate: int = 16000,
        max_audio_length: float = 30.0,
        max_utterances_per_recording: int = 100,
        include_audio: bool = True,
        include_lexical: bool = True,
        normalize_features: bool = True,
        filter_min_utterances: int = 1
    ):
        """
        Args:
            csv_path: Path to CSV file with utterance-level features
            audio_dir: Directory containing audio files (optional)
            sample_rate: Audio sample rate
            max_audio_length: Maximum audio length in seconds
            max_utterances_per_recording: Maximum utterances per recording
            include_audio: Whether to load audio features
            include_lexical: Whether to load lexical features
            normalize_features: Whether to normalize numerical features
            filter_min_utterances: Minimum utterances required per recording
        """
        self.csv_path = csv_path
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.max_utterances_per_recording = max_utterances_per_recording
        self.include_audio = include_audio
        self.include_lexical = include_lexical
        self.normalize_features = normalize_features
        self.filter_min_utterances = filter_min_utterances
        
        # Load and process data
        self._load_data()
        self._group_by_recording()
        
        # Feature dimensions
        self.embedding_dim = EMB_DIM  # 384
        self.handcrafted_dim = len(HANDCRAFTED_KEYS) + 1  # 15 + semantic_coherence = 16
        self.lexical_dim = self.embedding_dim + self.handcrafted_dim  # 400 total
        
        print(f"Loaded {len(self.recordings)} recordings with {self.total_utterances} total utterances")
        print(f"Average utterances per recording: {self.total_utterances / len(self.recordings):.1f}")
        
    def _load_data(self):
        """Load CSV data and validate structure."""
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        self.df = pd.read_csv(self.csv_path)
        
        # Validate required columns
        required_cols = ['recording_id', 'utterance_id', 'cognitive_decline']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Validate feature columns
        expected_feature_cols = set(ALL_FEATURE_NAMES)
        actual_feature_cols = set(self.df.columns[3:])  # Skip first 3 metadata columns
        
        if expected_feature_cols != actual_feature_cols:
            missing = expected_feature_cols - actual_feature_cols
            extra = actual_feature_cols - expected_feature_cols
            if missing:
                warnings.warn(f"Missing feature columns: {missing}")
            if extra:
                warnings.warn(f"Extra columns found: {extra}")
        
        # Extract feature columns
        self.feature_cols = [col for col in ALL_FEATURE_NAMES if col in self.df.columns]
        self.embedding_cols = [f"emb_{i}" for i in range(EMB_DIM) if f"emb_{i}" in self.df.columns]
        self.handcrafted_cols = [col for col in HANDCRAFTED_KEYS + ['semantic_coherence'] if col in self.df.columns]
        
        print(f"Found {len(self.feature_cols)} feature columns")
        print(f"  - {len(self.embedding_cols)} embedding features")
        print(f"  - {len(self.handcrafted_cols)} handcrafted features")
        
        # Handle missing values
        if self.df[self.feature_cols].isnull().any().any():
            warnings.warn("Found NaN values in features, filling with 0")
            self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)
            
        # Normalize features if requested
        if self.normalize_features:
            self._normalize_features()
            
    def _normalize_features(self):
        """Normalize numerical features to zero mean, unit variance."""
        from sklearn.preprocessing import StandardScaler
        
        # Normalize embeddings
        if self.embedding_cols:
            emb_scaler = StandardScaler()
            self.df[self.embedding_cols] = emb_scaler.fit_transform(self.df[self.embedding_cols])
            self.embedding_scaler = emb_scaler
        
        # Normalize handcrafted features (excluding binary features like has_filler)
        numeric_handcrafted = [col for col in self.handcrafted_cols if col not in ['has_filler']]
        if numeric_handcrafted:
            hc_scaler = StandardScaler()
            self.df[numeric_handcrafted] = hc_scaler.fit_transform(self.df[numeric_handcrafted])
            self.handcrafted_scaler = hc_scaler
        
        print("Applied feature normalization")
        
    def _group_by_recording(self):
        """Group utterances by recording_id and create recording-level dataset."""
        # Group by recording_id
        grouped = self.df.groupby('recording_id')
        
        self.recordings = []
        self.total_utterances = 0
        
        for recording_id, group in grouped:
            # Sort by utterance_id to maintain temporal order
            group = group.sort_values('utterance_id')
            
            # Filter recordings with too few utterances
            if len(group) < self.filter_min_utterances:
                continue
                
            # Limit utterances per recording
            if len(group) > self.max_utterances_per_recording:
                group = group.head(self.max_utterances_per_recording)
                warnings.warn(f"Truncated recording {recording_id} to {self.max_utterances_per_recording} utterances")
            
            # Extract features and label
            features = group[self.feature_cols].values.astype(np.float32)
            label = group['cognitive_decline'].iloc[0]  # Same label for all utterances in recording
            
            # Verify consistent labels within recording
            if not (group['cognitive_decline'] == label).all():
                warnings.warn(f"Inconsistent labels within recording {recording_id}")
            
            recording_data = {
                'recording_id': recording_id,
                'features': features,  # Shape: (num_utterances, feature_dim)
                'num_utterances': len(group),
                'label': int(label) if label != -1 else -1,
                'utterance_ids': group['utterance_id'].tolist()
            }
            
            self.recordings.append(recording_data)
            self.total_utterances += len(group)
            
        if not self.recordings:
            raise ValueError("No valid recordings found after filtering")
            
        # Create label distribution info
        valid_labels = [r['label'] for r in self.recordings if r['label'] != -1]
        if valid_labels:
            self.label_counts = {0: valid_labels.count(0), 1: valid_labels.count(1)}
            self.class_weights = {
                0: len(valid_labels) / (2 * self.label_counts[0]) if self.label_counts[0] > 0 else 1.0,
                1: len(valid_labels) / (2 * self.label_counts[1]) if self.label_counts[1] > 0 else 1.0
            }
            print(f"Label distribution: Control={self.label_counts[0]}, Dementia={self.label_counts[1]}")
        else:
            self.label_counts = {}
            self.class_weights = {}
            
    def __len__(self) -> int:
        return len(self.recordings)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a recording-level sample.
        
        Returns:
            Dict containing:
            - 'lexical_features': (num_utterances, lexical_dim) tensor
            - 'acoustic_features': (num_utterances, acoustic_dim) tensor (if audio available)
            - 'label': scalar label
            - 'recording_id': string ID
            - 'num_utterances': number of utterances
        """
        recording = self.recordings[idx]
        
        sample = {
            'recording_id': recording['recording_id'],
            'num_utterances': recording['num_utterances'],
            'label': recording['label'],
        }
        
        if self.include_lexical:
            # Split features into embeddings and handcrafted
            features = recording['features']  # (num_utterances, total_feature_dim)
            
            if self.embedding_cols:
                emb_features = features[:, :len(self.embedding_cols)]  # (num_utterances, 384)
            else:
                emb_features = np.zeros((recording['num_utterances'], self.embedding_dim), dtype=np.float32)
                
            if self.handcrafted_cols:
                hc_start_idx = len(self.embedding_cols)
                hc_features = features[:, hc_start_idx:hc_start_idx + len(self.handcrafted_cols)]
            else:
                hc_features = np.zeros((recording['num_utterances'], self.handcrafted_dim), dtype=np.float32)
            
            # Combine embeddings and handcrafted features
            lexical_features = np.concatenate([emb_features, hc_features], axis=1)
            sample['lexical_features'] = torch.from_numpy(lexical_features)
            
        if self.include_audio and self.audio_dir:
            # For now, create dummy acoustic features
            # In practice, you would load corresponding audio segments
            acoustic_features = torch.zeros(recording['num_utterances'], 768)  # WavLM dim
            sample['acoustic_features'] = acoustic_features
            
        return sample
        
    def get_class_weights(self) -> Dict[int, float]:
        """Get class weights for handling imbalanced data."""
        return self.class_weights.copy()
        
    def get_recording_stats(self) -> Dict[str, Any]:
        """Get statistics about recordings and utterances."""
        utterance_counts = [r['num_utterances'] for r in self.recordings]
        return {
            'num_recordings': len(self.recordings),
            'total_utterances': self.total_utterances,
            'avg_utterances_per_recording': np.mean(utterance_counts),
            'std_utterances_per_recording': np.std(utterance_counts),
            'min_utterances': min(utterance_counts),
            'max_utterances': max(utterance_counts),
            'label_distribution': self.label_counts
        }


def collate_utterance_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-length utterance sequences.
    
    Pads sequences to the same length within batch and creates attention masks.
    
    Args:
        batch: List of samples from UtteranceDataset
        
    Returns:
        Dict with batched and padded tensors:
        - 'lexical_features': (batch_size, max_seq_len, lexical_dim)
        - 'acoustic_features': (batch_size, max_seq_len, acoustic_dim)  
        - 'attention_mask': (batch_size, max_seq_len) bool mask
        - 'labels': (batch_size,) labels
        - 'num_utterances': (batch_size,) sequence lengths
        - 'recording_ids': List of recording IDs
    """
    batch_size = len(batch)
    max_seq_len = max(sample['num_utterances'] for sample in batch)
    
    # Extract data
    labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.long)
    num_utterances = torch.tensor([sample['num_utterances'] for sample in batch], dtype=torch.long)
    recording_ids = [sample['recording_id'] for sample in batch]
    
    # Create attention mask
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    for i, sample in enumerate(batch):
        attention_mask[i, :sample['num_utterances']] = True
    
    collated = {
        'labels': labels,
        'num_utterances': num_utterances,
        'recording_ids': recording_ids,
        'attention_mask': attention_mask
    }
    
    # Pad lexical features
    if 'lexical_features' in batch[0]:
        lexical_sequences = [sample['lexical_features'] for sample in batch]
        lexical_features = pad_sequence(lexical_sequences, batch_first=True, padding_value=0.0)
        collated['lexical_features'] = lexical_features
    
    # Pad acoustic features  
    if 'acoustic_features' in batch[0]:
        acoustic_sequences = [sample['acoustic_features'] for sample in batch]
        acoustic_features = pad_sequence(acoustic_sequences, batch_first=True, padding_value=0.0)
        collated['acoustic_features'] = acoustic_features
        
    return collated


def create_data_loaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 16,
    num_workers: int = 4,
    audio_dir: Optional[str] = None,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV  
        test_csv: Path to test CSV
        batch_size: Batch size
        num_workers: Number of data loading workers
        audio_dir: Audio directory (optional)
        **dataset_kwargs: Additional dataset arguments
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = UtteranceDataset(train_csv, audio_dir=audio_dir, **dataset_kwargs)
    val_dataset = UtteranceDataset(val_csv, audio_dir=audio_dir, **dataset_kwargs)
    test_dataset = UtteranceDataset(test_csv, audio_dir=audio_dir, **dataset_kwargs)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_utterance_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_utterance_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_utterance_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


# Utility functions
def analyze_dataset(csv_path: str) -> None:
    """Analyze and print statistics about a dataset."""
    dataset = UtteranceDataset(csv_path, include_audio=False)
    stats = dataset.get_recording_stats()
    
    print(f"\nDataset Analysis: {csv_path}")
    print("=" * 50)
    print(f"Recordings: {stats['num_recordings']}")
    print(f"Total utterances: {stats['total_utterances']}")
    print(f"Avg utterances/recording: {stats['avg_utterances_per_recording']:.1f} ± {stats['std_utterances_per_recording']:.1f}")
    print(f"Utterance range: {stats['min_utterances']} - {stats['max_utterances']}")
    print(f"Label distribution: {stats['label_distribution']}")
    print(f"Class weights: {dataset.get_class_weights()}")


if __name__ == "__main__":
    # Test the dataset
    import argparse
    
    parser = argparse.ArgumentParser(description="Test utterance dataset")
    parser.add_argument("--csv_path", required=True, help="Path to CSV file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    args = parser.parse_args()
    
    print("Testing UtteranceDataset...")
    analyze_dataset(args.csv_path)
    
    # Test data loading
    dataset = UtteranceDataset(args.csv_path, include_audio=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_utterance_batch)
    
    print(f"\nTesting DataLoader with batch_size={args.batch_size}...")
    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  Lexical features shape: {batch['lexical_features'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Labels: {batch['labels']}")
        print(f"  Num utterances: {batch['num_utterances']}")
        if i >= 2:  # Only show first few batches
            break
    
    print("Dataset test completed!")