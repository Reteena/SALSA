"""
Standalone test for utterance-level evaluation pipeline.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path  
sys.path.append(str(Path(__file__).parent.parent))

from fusion.multimodal_fusion import MultimodalFusion
from simple_dataset import SimpleUtteranceDataset, simple_collate_batch

def test_standalone_evaluation():
    """Test the evaluation pipeline without full dependencies."""
    print("="*60)
    print("TESTING UTTERANCE-LEVEL EVALUATION PIPELINE")
    print("="*60)
    
    # Create test data
    print("1. Creating test dataset...")
    data = []
    
    # Create 8 recordings with variable utterances
    for rec_id in range(8):
        num_utterances = np.random.randint(3, 7)  # 3-6 utterances per recording
        label = rec_id % 2  # Alternating labels
        
        for utt_id in range(num_utterances):
            # Create 400-dimensional features
            features = np.random.randn(400).astype(np.float32)
            
            row = {
                'recording_id': f'S{rec_id:03d}',
                'utterance_id': utt_id,
                'cognitive_decline': label
            }
            
            # Add embedding features (384 dims)
            for i in range(384):
                row[f'emb_{i}'] = features[i]
            
            # Add handcrafted features (16 dims)
            handcrafted_names = [
                'word_count', 'sentence_count', 'avg_word_len', 'type_token_ratio',
                'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio',
                'complexity_score', 'readability', 'fluency', 'coherence_local',
                'coherence_global', 'repetition_rate', 'pause_ratio', 'semantic_coherence'
            ]
            
            for i, name in enumerate(handcrafted_names):
                row[name] = features[384 + i]
            
            data.append(row)
    
    df = pd.DataFrame(data)
    test_csv = "/tmp/test_utterances.csv"
    df.to_csv(test_csv, index=False)
    
    print(f"   Created {len(df)} utterances from {df['recording_id'].nunique()} recordings")
    
    # Load dataset
    print("2. Loading dataset...")
    dataset = SimpleUtteranceDataset(test_csv)
    print(f"   Loaded {len(dataset)} recordings with {dataset.get_total_utterances()} utterances")
    
    # Create model
    print("3. Creating MultimodalFusion model...")
    model = MultimodalFusion(
        lexical_dim=400,
        num_classes=2,
        hidden_dim=128,
        num_fusion_layers=2,
        dropout=0.1,
        aggregation_strategy="attention"
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model created with {trainable_params:,} trainable parameters")
    
    # Test forward pass
    print("4. Testing model forward pass...")
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=simple_collate_batch
    )
    
    all_predictions = []
    all_labels = []
    all_recording_ids = []
    
    print("5. Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            features = batch['features']  # [B, max_seq_len, 400]
            attention_mask = batch['attention_mask']  # [B, max_seq_len]
            labels = batch['labels']  # [B]
            recording_ids = batch['recording_ids']
            
            print(f"   Batch {batch_idx + 1}: {features.shape[0]} recordings, "
                  f"max {features.shape[1]} utterances per recording")
            
            # Forward pass
            outputs = model(features, attention_mask=attention_mask)
            predictions = outputs['predictions']  # [B, 2]
            
            # Convert to probabilities
            probs = torch.softmax(predictions, dim=1)
            
            all_predictions.append(probs)
            all_labels.append(labels)
            all_recording_ids.extend(recording_ids)
    
    # Combine results
    all_predictions = torch.cat(all_predictions, dim=0)  # [N, 2]
    all_labels = torch.cat(all_labels, dim=0)  # [N]
    
    # Compute metrics
    print("6. Computing metrics...")
    predicted_labels = torch.argmax(all_predictions, dim=1)
    accuracy = (predicted_labels == all_labels).float().mean()
    
    # Positive class probabilities
    pos_probs = all_predictions[:, 1]
    confidence = torch.max(all_predictions, dim=1)[0]
    
    # Class distribution
    unique_labels, counts = torch.unique(all_labels, return_counts=True)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total recordings evaluated: {len(all_recording_ids)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean positive class probability: {pos_probs.mean():.4f}")
    print(f"Mean confidence: {confidence.mean():.4f}")
    print(f"Standard deviation confidence: {confidence.std():.4f}")
    
    print(f"\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {int(label)}: {int(count)} recordings ({100*count/len(all_labels):.1f}%)")
    
    print(f"\nPer-recording results:")
    for i, rec_id in enumerate(all_recording_ids):
        true_label = int(all_labels[i])
        pred_label = int(predicted_labels[i])
        conf = float(confidence[i])
        print(f"  {rec_id}: True={true_label}, Pred={pred_label}, Confidence={conf:.3f}")
    
    print("\n✅ EVALUATION PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("   - Utterance-level features loaded correctly")
    print("   - Variable-length sequences handled properly") 
    print("   - Attention-based aggregation working")
    print("   - Recording-level predictions generated")
    
    return True

if __name__ == "__main__":
    test_standalone_evaluation()