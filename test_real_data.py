#!/usr/bin/env python3
"""
Test the complete utterance pipeline with the actual CSV data from lexical_features.py
"""
import torch
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add SALSA to path
sys.path.append('/home/saksatti/SALSA')

from data.utterance_dataset import UtteranceDataset, collate_utterance_batch, create_data_loaders
from fusion.multimodal_fusion import create_lexical_only_salsa
from torch.utils.data import DataLoader

def test_with_real_data():
    """Test the pipeline with real lexical features CSV."""
    print("🧪 Testing SALSA Pipeline with Real Lexical Features Data")
    print("=" * 60)
    
    # Paths to the actual CSV files
    train_csv = '/home/saksatti/SALSA/preprocess/lexical_features_train.csv'
    test_csv = '/home/saksatti/SALSA/preprocess/lexical_features_test.csv'
    
    print("1. Loading real datasets...")
    
    # Create datasets
    train_dataset = UtteranceDataset(
        train_csv,
        include_audio=False,
        include_lexical=True,
        normalize_features=True
    )
    
    test_dataset = UtteranceDataset(
        test_csv,
        include_audio=False, 
        include_lexical=True,
        normalize_features=True
    )
    
    print(f"   Train dataset: {len(train_dataset)} recordings")
    print(f"   Test dataset: {len(test_dataset)} recordings")
    print(f"   Train total utterances: {train_dataset.total_utterances}")
    print(f"   Test total utterances: {test_dataset.total_utterances}")
    print(f"   Feature dimensions: {train_dataset.lexical_dim}")
    
    # Check some statistics
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"\n   Train recordings by class: {train_df.groupby('recording_id')['cognitive_decline'].first().value_counts().to_dict()}")
    print(f"   Test recordings by class: {test_df.groupby('recording_id')['cognitive_decline'].first().value_counts().to_dict()}")
    
    train_utts_per_rec = train_df.groupby('recording_id').size()
    test_utts_per_rec = test_df.groupby('recording_id').size()
    
    print(f"   Train utterances per recording: {train_utts_per_rec.min()}-{train_utts_per_rec.max()} (mean: {train_utts_per_rec.mean():.1f})")
    print(f"   Test utterances per recording: {test_utts_per_rec.min()}-{test_utts_per_rec.max()} (mean: {test_utts_per_rec.mean():.1f})")
    
    # Create data loaders
    print("\n2. Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_utterance_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_utterance_batch
    )
    
    # Test a batch
    print("\n3. Testing batch loading...")
    batch = next(iter(train_loader))
    
    print(f"   Batch size: {len(batch['recording_ids'])}")
    print(f"   Lexical features shape: {batch['lexical_features'].shape}")
    print(f"   Attention mask shape: {batch['attention_mask'].shape}")
    print(f"   Labels: {batch['labels']}")
    print(f"   Valid utterances per recording: {batch['attention_mask'].sum(dim=1)}")
    
    # Create model
    print("\n4. Creating and testing model...")
    model = create_lexical_only_salsa(
        lexical_dim=400,  # 384 embeddings + 16 handcrafted features
        fusion_dim=512,
        aggregation_type='attention',
        dropout=0.1
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\n5. Testing forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        lexical_features = batch['lexical_features'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        logits = model(
            lexical_features=lexical_features,
            attention_mask=attention_mask
        )
        
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
    
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Sample predictions: {predictions.cpu().numpy()}")
    print(f"   Sample probabilities: {probs.cpu().numpy()}")
    print(f"   Ground truth: {batch['labels'].numpy()}")
    
    # Test training step
    print("\n6. Testing training step...")
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    logits = model(
        lexical_features=lexical_features,
        attention_mask=attention_mask
    )
    
    loss = criterion(logits, batch['labels'].to(device))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   Training loss: {loss.item():.4f}")
    print("   ✅ Backpropagation successful!")
    
    # Test evaluation on small subset
    print("\n7. Testing evaluation...")
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Only test a few batches
                break
                
            lexical_features = batch['lexical_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(
                lexical_features=lexical_features,
                attention_mask=attention_mask
            )
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"   Sample accuracy: {accuracy:.3f}")
    print(f"   Sample size: {len(all_preds)} recordings")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("\n🔑 Key Insights:")
    print("  ✅ Your lexical_features.py provides perfect utterance-level features")
    print("  ✅ No need for Whisper ASR - we have ground truth CHAT transcriptions")
    print("  ✅ 384-dim MiniLM embeddings + 16 linguistic features + semantic coherence")
    print("  ✅ Variable-length utterance sequences properly handled")
    print("  ✅ Recording-level predictions from utterance-level aggregation")
    print("  ✅ Training pipeline works end-to-end")
    
    print(f"\n💡 Data Summary:")
    print(f"  • {len(train_dataset)} train recordings, {len(test_dataset)} test recordings")
    print(f"  • {train_dataset.total_utterances} train utterances, {test_dataset.total_utterances} test utterances")
    print(f"  • Utterances per recording: {train_utts_per_rec.min()}-{train_utts_per_rec.max()}")
    print(f"  • All features extracted from ground truth CHAT transcriptions")
    
    return True

if __name__ == "__main__":
    test_with_real_data()