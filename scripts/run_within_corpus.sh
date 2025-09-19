#!/bin/bash

# SALSA Within-Corpus Training - Simple training on your ADReSS data
# Uses train/test split from your CSV files

set -e  # Exit on any error

# Default parameters
TRAIN_CSV="./preprocess/lexical_features_train.csv"
TEST_CSV="./preprocess/lexical_features_test.csv"
OUTPUT_DIR="./results"
DEVICE="auto"
BATCH_SIZE=16
MAX_EPOCHS=100
PATIENCE=15
SEED=42

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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --train-csv FILE      Training CSV file (default: ./preprocess/lexical_features_train.csv)"
            echo "  --test-csv FILE       Test CSV file (default: ./preprocess/lexical_features_test.csv)"
            echo "  --output-dir DIR      Output directory (default: ./results)"
            echo "  --device DEVICE       Device to use: cuda, cpu, or auto (default: auto)"
            echo "  --batch-size SIZE     Batch size (default: 16)"
            echo "  --max-epochs N        Maximum epochs (default: 100)"
            echo "  --patience N          Early stopping patience (default: 15)"
            echo "  --seed N              Random seed (default: 42)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "SALSA Within-Corpus Training on ADReSS"
echo "=========================================="
echo "Training CSV: $TRAIN_CSV"
echo "Test CSV: $TEST_CSV"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "Max epochs: $MAX_EPOCHS"
echo "Patience: $PATIENCE"
echo "Random seed: $SEED"
echo ""

# Validate input files
if [[ ! -f "$TRAIN_CSV" ]]; then
    echo "ERROR: Training CSV file not found: $TRAIN_CSV"
    exit 1
fi

if [[ ! -f "$TEST_CSV" ]]; then
    echo "ERROR: Test CSV file not found: $TEST_CSV"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Common arguments
COMMON_ARGS="--batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS --patience $PATIENCE --seed $SEED"
COMMON_ARGS="$COMMON_ARGS --train_csv $TRAIN_CSV --test_csv $TEST_CSV --output_dir $OUTPUT_DIR"

if [ "$DEVICE" != "auto" ]; then
    COMMON_ARGS="$COMMON_ARGS --device $DEVICE"
fi

echo "Running within-corpus experiments on your ADReSS data..."
echo ""

# Main lexical model with your CHAT features
echo "1. Training main lexical model..."
python train/train.py \
    --experiment_name "adress_lexical_main" \
    --model_type lexical_only \
    --lexical_dim 400 \
    --val_csv $TEST_CSV \
    $COMMON_ARGS

echo ""

# ERM training
echo "2. ERM training..."
python train/train_erm.py \
    --experiment_name "adress_erm" \
    --model_type lexical_only \
    --lexical_dim 400 \
    --val_csv $TEST_CSV \
    $COMMON_ARGS

echo ""

# GroupDRO training
echo "3. GroupDRO training..."
python train/train_groupdro.py \
    --experiment_name "adress_groupdro" \
    --model_type lexical_only \
    --lexical_dim 400 \
    --val_csv $TEST_CSV \
    $COMMON_ARGS

echo ""
echo "=========================================="
echo "Within-corpus experiments completed!"
echo "=========================================="
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Experiments:"
echo "- adress_lexical_main: Basic lexical model"
echo "- adress_erm: ERM training"
echo "- adress_groupdro: GroupDRO robust training"
echo ""
echo "Each contains results.json with performance metrics"
            echo "Options:"
            echo "  --data-dir DIR        Data directory (default: ./data)"
            echo "  --output-dir DIR      Output directory (default: ./results)"
            echo "  --device DEVICE       Device to use: cuda, cpu, or auto (default: auto)"
            echo "  --batch-size SIZE     Batch size (default: 16)"
            echo "  --max-epochs N        Maximum epochs (default: 100)"
            echo "  --patience N          Early stopping patience (default: 15)"
            echo "  --seed N              Random seed (default: 42)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "SALSA Within-Corpus Experiments"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "Max epochs: $MAX_EPOCHS"
echo "Patience: $PATIENCE"
echo "Random seed: $SEED"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/plots"

# Common arguments for all experiments
COMMON_ARGS="--data_dir $DATA_DIR --output_dir $OUTPUT_DIR --batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS --patience $PATIENCE --seed $SEED"

if [ "$DEVICE" != "auto" ]; then
    COMMON_ARGS="$COMMON_ARGS --device $DEVICE"
fi

# Dataset configurations
declare -A datasets
datasets["adress"]="ADReSS-2020"
datasets["adresso"]="ADReSSo-2021-2022"
datasets["pitt"]="DementiaBank-Pitt"

# Experiment configurations
declare -A experiments
experiments["baseline_egemaps"]="openSMILE eGeMAPS + Logistic Regression baseline"
experiments["baseline_wavlm_probe"]="Frozen WavLM + Linear Probe baseline"
experiments["erm_multimodal"]="Multimodal SALSA with ERM training"
experiments["groupdro_multimodal"]="Multimodal SALSA with GroupDRO training"
experiments["acoustic_only"]="Acoustic-only model"
experiments["lexical_only"]="Lexical-only model"

echo "Running within-corpus experiments..."
echo ""

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local exp_name=$2
    local exp_description="$3"
    local script_name=$4
    local extra_args="$5"
    
    echo "----------------------------------------"
    echo "Running: $exp_description"
    echo "Dataset: ${datasets[$dataset]}"
    echo "----------------------------------------"
    
    local full_exp_name="${dataset}_${exp_name}"
    local manifest_dir="$DATA_DIR/$dataset"
    local train_manifest="$manifest_dir/train_manifest.json"
    local val_manifest="$manifest_dir/val_manifest.json"
    local test_manifest="$manifest_dir/test_manifest.json"
    
    # Check if manifest files exist
    if [[ ! -f "$train_manifest" ]]; then
        echo "WARNING: Training manifest not found: $train_manifest"
        echo "Skipping $full_exp_name"
        return 1
    fi
    
    if [[ ! -f "$val_manifest" ]]; then
        echo "WARNING: Validation manifest not found: $val_manifest"
        echo "Skipping $full_exp_name"
        return 1
    fi
    
    # Run experiment
    local log_file="$OUTPUT_DIR/logs/${full_exp_name}.log"
    echo "Experiment: $full_exp_name" > "$log_file"
    echo "Started at: $(date)" >> "$log_file"
    echo "" >> "$log_file"
    
    if python "$script_name" \
        --train_manifest "$train_manifest" \
        --val_manifest "$val_manifest" \
        --experiment_name "$full_exp_name" \
        --log_dir "$OUTPUT_DIR/logs" \
        --checkpoint_dir "$OUTPUT_DIR/checkpoints" \
        $COMMON_ARGS $extra_args 2>&1 | tee -a "$log_file"; then
        
        echo "✓ Completed: $full_exp_name"
        echo "Finished at: $(date)" >> "$log_file"
    else
        echo "✗ Failed: $full_exp_name"
        echo "FAILED at: $(date)" >> "$log_file"
        return 1
    fi
    
    echo ""
}

# Function to run baseline experiments
run_baseline_experiments() {
    local dataset=$1
    
    echo "Running baseline experiments for ${datasets[$dataset]}..."
    
    # openSMILE eGeMAPS baseline
    if run_experiment "$dataset" "baseline_egemaps" "${experiments[baseline_egemaps]}" \
        "baselines/run_opensmile_baseline.py" "--feature_set egemaps --classifier_type logistic"; then
        echo "Baseline eGeMAPS completed for $dataset"
    fi
    
    # Frozen WavLM baseline
    if run_experiment "$dataset" "baseline_wavlm_probe" "${experiments[baseline_wavlm_probe]}" \
        "baselines/run_wavlm_baseline.py" "--pooling_strategy attention"; then
        echo "Baseline WavLM probe completed for $dataset"
    fi
}

# Function to run main SALSA experiments
run_salsa_experiments() {
    local dataset=$1
    
    echo "Running SALSA experiments for ${datasets[$dataset]}..."
    
    # ERM training (multimodal)
    if run_experiment "$dataset" "erm_multimodal" "${experiments[erm_multimodal]}" \
        "train/train_erm.py" "--model_type multimodal"; then
        echo "ERM multimodal completed for $dataset"
    fi
    
    # GroupDRO training (multimodal)
    if run_experiment "$dataset" "groupdro_multimodal" "${experiments[groupdro_multimodal]}" \
        "train/train_groupdro.py" "--model_type multimodal"; then
        echo "GroupDRO multimodal completed for $dataset"
    fi
    
    # Acoustic-only model
    if run_experiment "$dataset" "acoustic_only" "${experiments[acoustic_only]}" \
        "train/train_erm.py" "--model_type acoustic_only"; then
        echo "Acoustic-only completed for $dataset"
    fi
    
    # Lexical-only model  
    if run_experiment "$dataset" "lexical_only" "${experiments[lexical_only]}" \
        "train/train_erm.py" "--model_type lexical_only"; then
        echo "Lexical-only completed for $dataset"
    fi
}

# Main execution
echo "Starting within-corpus experiments..."
start_time=$(date +%s)

# Run experiments for each dataset
for dataset in "${!datasets[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing dataset: ${datasets[$dataset]}"
    echo "=========================================="
    
    # Check if dataset directory exists
    if [[ ! -d "$DATA_DIR/$dataset" ]]; then
        echo "WARNING: Dataset directory not found: $DATA_DIR/$dataset"
        echo "Skipping ${datasets[$dataset]}"
        continue
    fi
    
    # Run baseline experiments
    run_baseline_experiments "$dataset"
    
    # Run main SALSA experiments
    run_salsa_experiments "$dataset"
    
    echo "Completed all experiments for ${datasets[$dataset]}"
done

# Generate summary report
echo ""
echo "Generating summary report..."
python eval/generate_within_corpus_report.py \
    --results_dir "$OUTPUT_DIR" \
    --output_file "$OUTPUT_DIR/within_corpus_summary.html"

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "=========================================="
echo "Within-Corpus Experiments Completed!"
echo "=========================================="
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo "Results saved to: $OUTPUT_DIR"
echo "Summary report: $OUTPUT_DIR/within_corpus_summary.html"
echo ""
echo "Key files:"
echo "  - Logs: $OUTPUT_DIR/logs/"
echo "  - Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "  - Individual results: $OUTPUT_DIR/logs/*_results.pt"
echo ""
echo "Next steps:"
echo "  1. Review results in the summary report"
echo "  2. Run LOCO evaluation: ./scripts/run_LOCO.sh"
echo "  3. Run ablation studies: ./scripts/run_ablations.sh"