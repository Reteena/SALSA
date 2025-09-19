#!/bin/bash

# Simple script to train lexical-only SALSA model on ADReSS data
# This uses your 400-dimensional utterance features (384 embeddings + 16 handcrafted)

set -e  # Exit on any error

echo "========================================"
echo "SALSA Lexical-Only Training"
echo "========================================"

# Default paths and parameters
TRAIN_CSV="./preprocess/lexical_features_train.csv"
TEST_CSV="./preprocess/lexical_features_test.csv"
OUTPUT_DIR="./experiments"
DEVICE="auto"
BATCH_SIZE=16
MAX_EPOCHS=50
PATIENCE=10
SEED=42
EXPERIMENT_NAME="lexical_only_adress"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-csv)
            TRAIN_CSV="$2"
            shift 2
            ;;
        --test-csv)
            TEST_CSV="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --train-csv PATH      Training CSV file (default: $TRAIN_CSV)"
            echo "  --test-csv PATH       Test CSV file (default: $TEST_CSV)"
            echo "  --output-dir PATH     Output directory (default: $OUTPUT_DIR)"
            echo "  --device DEVICE       Device to use (default: $DEVICE)"
            echo "  --batch-size SIZE     Batch size (default: $BATCH_SIZE)"
            echo "  --max-epochs NUM      Max epochs (default: $MAX_EPOCHS)"
            echo "  --patience NUM        Early stopping patience (default: $PATIENCE)"
            echo "  --seed NUM            Random seed (default: $SEED)"
            echo "  --experiment-name NAME Experiment name (default: $EXPERIMENT_NAME)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "Training CSV: $TRAIN_CSV"
echo "Test CSV: $TEST_CSV"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "Max epochs: $MAX_EPOCHS"
echo "Patience: $PATIENCE"
echo "Random seed: $SEED"
echo "Experiment: $EXPERIMENT_NAME"
echo ""

# Validate input files exist
if [[ ! -f "$TRAIN_CSV" ]]; then
    echo "ERROR: Training CSV file not found: $TRAIN_CSV"
    echo "Make sure you've run lexical feature extraction first:"
    echo "  cd preprocess && python lexical_features.py"
    exit 1
fi

if [[ ! -f "$TEST_CSV" ]]; then
    echo "ERROR: Test CSV file not found: $TEST_CSV"
    echo "Make sure you've run lexical feature extraction first:"
    echo "  cd preprocess && python lexical_features.py"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Training lexical-only SALSA model..."
echo "Using 400-dimensional utterance features from CHAT transcripts"
echo "Features: 384 MiniLM embeddings + 16 handcrafted linguistic features"
echo ""

# Build command arguments
ARGS="--experiment_name $EXPERIMENT_NAME"
ARGS="$ARGS --model_type lexical_only"
ARGS="$ARGS --lexical_dim 400"
ARGS="$ARGS --train_csv $TRAIN_CSV"
ARGS="$ARGS --val_csv $TEST_CSV"
ARGS="$ARGS --test_csv $TEST_CSV"
ARGS="$ARGS --output_dir $OUTPUT_DIR"
ARGS="$ARGS --batch_size $BATCH_SIZE"
ARGS="$ARGS --max_epochs $MAX_EPOCHS"
ARGS="$ARGS --patience $PATIENCE"
ARGS="$ARGS --seed $SEED"

if [ "$DEVICE" != "auto" ]; then
    ARGS="$ARGS --device $DEVICE"
fi

# Run training
echo "Command: python train/train.py $ARGS"
echo ""

python train/train.py $ARGS

echo ""
echo "========================================"
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo "========================================"