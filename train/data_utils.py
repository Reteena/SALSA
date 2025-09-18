"""
Data utilities for SALSA project.
Handles dataset creation, data loading, and preprocessing.
"""
import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging


# Removed SALSADataset - using UtteranceDataset from data/utterance_dataset.py instead

def create_data_loaders_from_csv(train_csv: str,
                                test_csv: str,
                                batch_size: int = 16,
                                num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders from CSV files using UtteranceDataset.
    
    Args:
        train_csv: Training CSV file path
        test_csv: Test CSV file path  
        batch_size: Batch size
        num_workers: Number of data loader workers
        
    Returns:
        (train_loader, test_loader)
    """
    from data.utterance_dataset import UtteranceDataset
    
    # Create datasets
    train_dataset = UtteranceDataset(train_csv)
    test_dataset = UtteranceDataset(test_csv)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader


def create_manifest_file(data_dir: str, 
                        audio_files: List[str],
                        labels: List[int],
                        group_ids: List[int],
                        transcripts: Optional[List[str]] = None,
                        output_file: str = "manifest.json"):
    """
    Create a manifest file for the dataset.
    
    Args:
        data_dir: Base data directory
        audio_files: List of audio file paths (relative to data_dir)
        labels: List of labels (0 for healthy, 1 for dementia)
        group_ids: List of group IDs (dataset identifiers)
        transcripts: Optional list of transcripts
        output_file: Output manifest file path
    """
    samples = []
    
    for i, (audio_file, label, group_id) in enumerate(zip(audio_files, labels, group_ids)):
        sample = {
            'sample_id': i,
            'audio_path': audio_file,
            'label': label,
            'group_id': group_id
        }
        
        if transcripts is not None and i < len(transcripts):
            sample['transcript'] = transcripts[i]
        
        samples.append(sample)
    
    # Save manifest
    output_path = Path(data_dir) / output_file
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Created manifest file with {len(samples)} samples: {output_path}")


def create_data_loaders(data_dir: str,
                       train_manifest: str,
                       val_manifest: str,
                       test_manifest: Optional[str] = None,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       use_precomputed_features: bool = True) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    DEPRECATED: Use create_data_loaders_from_csv instead.
    This function is kept for backwards compatibility but SALSADataset has been removed.
    """
    raise NotImplementedError("SALSADataset has been removed. Use create_data_loaders_from_csv with CSV files instead.")


class ADReSSDataProcessor:
    """Processor for ADReSS dataset."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
    
    def create_manifests(self) -> Tuple[str, str, str]:
        """
        Create manifest files for ADReSS train/val/test splits.
        
        Returns:
            (train_manifest, val_manifest, test_manifest)
        """
        # Process training data
        train_dir = self.data_dir / 'train'
        train_samples = self._process_split(train_dir, group_id=0, split='train')
        
        # Split training data into train/val
        np.random.shuffle(train_samples)
        split_idx = int(0.8 * len(train_samples))
        actual_train = train_samples[:split_idx]
        val_samples = train_samples[split_idx:]
        
        # Process test data
        test_dir = self.data_dir / 'test'
        test_samples = self._process_split(test_dir, group_id=0, split='test')
        
        # Save manifests
        train_manifest = self.data_dir / 'train_manifest.json'
        val_manifest = self.data_dir / 'val_manifest.json'
        test_manifest = self.data_dir / 'test_manifest.json'
        
        with open(train_manifest, 'w') as f:
            json.dump(actual_train, f, indent=2)
        
        with open(val_manifest, 'w') as f:
            json.dump(val_samples, f, indent=2)
        
        with open(test_manifest, 'w') as f:
            json.dump(test_samples, f, indent=2)
        
        self.logger.info(f"Created manifests: train={len(actual_train)}, val={len(val_samples)}, test={len(test_samples)}")
        
        return str(train_manifest), str(val_manifest), str(test_manifest)
    
    def _process_split(self, split_dir: Path, group_id: int, split: str) -> List[Dict]:
        """Process a single data split."""
        samples = []
        
        # Load metadata
        if split == 'train':
            cc_meta = split_dir / 'cc_meta_data.txt'
            cd_meta = split_dir / 'cd_meta_data.txt'
            
            # Process healthy controls (CC)
            if cc_meta.exists():
                cc_samples = self._parse_metadata(cc_meta, label=0, group_id=group_id)
                samples.extend(cc_samples)
            
            # Process dementia patients (CD)  
            if cd_meta.exists():
                cd_samples = self._parse_metadata(cd_meta, label=1, group_id=group_id)
                samples.extend(cd_samples)
                
        else:  # test
            meta_file = split_dir / 'meta_data_test.txt'
            if meta_file.exists():
                test_samples = self._parse_metadata(meta_file, label=None, group_id=group_id)
                samples.extend(test_samples)
        
        return samples
    
    def _parse_metadata(self, meta_file: Path, label: Optional[int], group_id: int) -> List[Dict]:
        """Parse metadata file."""
        samples = []
        
        with open(meta_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    file_id = parts[0]
                    
                    # Construct audio path
                    audio_path = f"Full_wave_enhanced_audio/{file_id}.wav"
                    
                    sample = {
                        'sample_id': file_id,
                        'audio_path': audio_path,
                        'group_id': group_id
                    }
                    
                    if label is not None:
                        sample['label'] = label
                    else:
                        # For test set, try to infer label from file patterns or use placeholder
                        sample['label'] = 0  # Placeholder - should be updated with actual test labels
                    
                    samples.append(sample)
        
        return samples


def precompute_features(data_dir: str,
                       manifest_file: str,
                       acoustic_model,
                       lexical_model,
                       batch_size: int = 8,
                       device: str = 'cuda'):
    """
    DEPRECATED: Feature precomputation not needed since lexical features are pre-extracted from CHAT files.
    Acoustic features can be extracted on-the-fly if needed.
    """
    raise NotImplementedError("Feature precomputation deprecated. Use lexical_features.py for CHAT-based features.")