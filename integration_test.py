#!/usr/bin/env python3
"""
Integration example showing how to use SALSA with your existing lexical_features.csv files.
This script demonstrates the proper workflow from your utterance-level CSV to trained model.
"""
import argparse
import sys
import os
from pathlib import Path

# Add SALSA to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.utterance_dataset import UtteranceDataset, analyze_dataset
from fusion.multimodal_fusion import create_lexical_only_salsa
from train.train import main as train_main


def analyze_existing_data(csv_path: str):
    """Analyze existing lexical features CSV to understand the data structure."""
    print(f"\nAnalyzing existing data: {csv_path}")
    print("=" * 50)
    
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return False
    
    try:
        analyze_dataset(csv_path)
        return True
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return False


def create_train_val_split(csv_path: str, train_ratio: float = 0.8):
    """Create train/val split from a single CSV file."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    print(f"\nCreating train/val split from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Get unique recordings with their labels
    recording_info = df.groupby('recording_id')['cognitive_decline'].first().reset_index()
    
    # Split by recordings to avoid data leakage
    train_recordings, val_recordings = train_test_split(
        recording_info['recording_id'].tolist(),
        test_size=1-train_ratio,
        stratify=recording_info['cognitive_decline'],
        random_state=42
    )
    
    # Create train and val DataFrames
    train_df = df[df['recording_id'].isin(train_recordings)]
    val_df = df[df['recording_id'].isin(val_recordings)]
    
    # Save splits
    base_path = Path(csv_path).parent
    train_path = base_path / f"{Path(csv_path).stem}_train.csv"
    val_path = base_path / f"{Path(csv_path).stem}_val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"✓ Train split: {len(train_df)} utterances from {len(train_recordings)} recordings -> {train_path}")
    print(f"✓ Val split: {len(val_df)} utterances from {len(val_recordings)} recordings -> {val_path}")
    
    return str(train_path), str(val_path)


def quick_training_example(train_csv: str, val_csv: str, test_csv: str):
    """Run a quick training example to verify everything works."""
    print("\nRunning quick training example...")
    print("=" * 50)
    
    # Prepare arguments for training script
    import sys
    
    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    try:
        # Set up arguments for training script
        sys.argv = [
            'train.py',
            '--train_csv', train_csv,
            '--val_csv', val_csv,
            '--test_csv', test_csv,
            '--experiment_name', 'test_utterance_pipeline',
            '--model_type', 'lexical_only',
            '--batch_size', '4',
            '--max_epochs', '3',  # Quick test
            '--learning_rate', '1e-3',
            '--output_dir', './test_experiments',
            '--aggregation_type', 'attention',
            '--fusion_dim', '256'  # Smaller for quick test
        ]
        
        # Run training
        train_main()
        
        print("✓ Quick training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(description="Integration test for SALSA utterance pipeline")
    parser.add_argument('--train_csv', help='Path to training CSV (lexical_features_train.csv)')
    parser.add_argument('--test_csv', help='Path to test CSV (lexical_features_test.csv)')
    parser.add_argument('--single_csv', help='Single CSV to split into train/val')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze data, do not train')
    parser.add_argument('--quick_train', action='store_true', help='Run quick training test')
    
    args = parser.parse_args()
    
    print("SALSA Integration Test")
    print("=" * 60)
    print("This script tests the utterance-level pipeline with your existing data.")
    
    if args.single_csv:
        # Single CSV mode - split into train/val
        if not analyze_existing_data(args.single_csv):
            return
        
        if not args.analyze_only:
            train_csv, val_csv = create_train_val_split(args.single_csv)
            if args.quick_train:
                quick_training_example(train_csv, val_csv, val_csv)  # Use val as test for quick test
    
    elif args.train_csv and args.test_csv:
        # Separate train/test CSV mode
        if not analyze_existing_data(args.train_csv):
            return
        
        if not analyze_existing_data(args.test_csv):
            return
        
        if not args.analyze_only:
            # Create val split from train
            train_split, val_split = create_train_val_split(args.train_csv, train_ratio=0.8)
            
            if args.quick_train:
                quick_training_example(train_split, val_split, args.test_csv)
    
    else:
        print("\nUsage examples:")
        print("1. Analyze existing files:")
        print("   python integration_test.py --train_csv preprocess/lexical_features_train.csv --test_csv preprocess/lexical_features_test.csv --analyze_only")
        print("")
        print("2. Quick training test with existing files:")
        print("   python integration_test.py --train_csv preprocess/lexical_features_train.csv --test_csv preprocess/lexical_features_test.csv --quick_train")
        print("")
        print("3. Split single CSV and test:")
        print("   python integration_test.py --single_csv preprocess/lexical_features_train.csv --quick_train")
        print("")
        print("Expected CSV format (from your lexical_features.py):")
        print("- recording_id: Recording identifier (e.g., 'S001')")
        print("- utterance_id: Utterance identifier (e.g., 'S001_0')")  
        print("- cognitive_decline: Label (0 or 1)")
        print("- emb_0 to emb_383: MiniLM embeddings (384 features)")
        print("- Handcrafted features: utt_len, has_filler, POS counts, etc. (15 features)")
        print("- semantic_coherence: Cosine similarity with previous utterance")
        print("Total: 400 features per utterance")


if __name__ == "__main__":
    main()