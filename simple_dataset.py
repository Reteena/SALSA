"""
Simplified utterance dataset for testing without full dependencies.
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional

class SimpleUtteranceDataset(Dataset):
    """Simple version of UtteranceDataset for testing."""
    
    def __init__(self, csv_path: str):
        """Initialize dataset from CSV file."""
        self.df = pd.read_csv(csv_path)
        
        # Group by recording_id
        self.recordings = self.df.groupby('recording_id').apply(
            lambda x: x.sort_values('utterance_id').reset_index(drop=True)
        ).reset_index(level=0, drop=True)
        
        self.recording_ids = list(self.df['recording_id'].unique())
        self.recording_groups = {
            rec_id: self.df[self.df['recording_id'] == rec_id].sort_values('utterance_id')
            for rec_id in self.recording_ids
        }
        
    def __len__(self) -> int:
        return len(self.recording_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        recording_id = self.recording_ids[idx]
        recording_data = self.recording_groups[recording_id]
        
        # Extract features (assume columns 3 onwards are features)
        feature_cols = [col for col in recording_data.columns 
                       if col not in ['recording_id', 'utterance_id', 'cognitive_decline']]
        
        features = torch.tensor(recording_data[feature_cols].values, dtype=torch.float32)
        label = torch.tensor(recording_data['cognitive_decline'].iloc[0], dtype=torch.long)
        
        return {
            'recording_id': recording_id,
            'features': features,  # [num_utterances, feature_dim]
            'label': label,
            'num_utterances': len(recording_data)
        }
    
    def get_total_utterances(self) -> int:
        return len(self.df)

def simple_collate_batch(batch: List[Dict]) -> Dict:
    """Simple collate function for utterance batches."""
    max_seq_len = max(item['num_utterances'] for item in batch)
    batch_size = len(batch)
    feature_dim = batch[0]['features'].shape[1]
    
    # Create padded tensors
    padded_features = torch.zeros(batch_size, max_seq_len, feature_dim)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)
    recording_ids = []
    
    for i, item in enumerate(batch):
        seq_len = item['num_utterances']
        padded_features[i, :seq_len] = item['features']
        attention_mask[i, :seq_len] = True
        labels[i] = item['label']
        recording_ids.append(item['recording_id'])
    
    return {
        'features': padded_features,
        'attention_mask': attention_mask,
        'labels': labels,
        'recording_ids': recording_ids
    }