#!/bin/bash

# SALSA Training Runner - Simple script for your ADReSS utterance-level data
# Uses your extracted lexical features CSV files

set -e  # Exit on any error

# Default parameters
TRAIN_CSV="./preprocess/lexical_features_train.csv"
TEST_CSV="./preprocess/lexical_features_test.csv"
OUTPUT_DIR="./experiments"
DEVICE="auto"
BATCH_SIZE=16
MAX_EPOCHS=50
PATIENCE=10
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
            echo "  --output-dir DIR      Output directory (default: ./experiments)"
            echo "  --device DEVICE       Device to use: cuda, cpu, or auto (default: auto)"
            echo "  --batch-size SIZE     Batch size (default: 16)"
            echo "  --max-epochs N        Maximum epochs (default: 50)"
            echo "  --patience N          Early stopping patience (default: 10)"
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
echo "SALSA Training on ADReSS Utterance Data"
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

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Common arguments for all experiments
COMMON_ARGS="--batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS --patience $PATIENCE --seed $SEED"
COMMON_ARGS="$COMMON_ARGS --train_csv $TRAIN_CSV"

if [ "$DEVICE" != "auto" ]; then
    COMMON_ARGS="$COMMON_ARGS --device $DEVICE"
fi

echo "Running experiments with your ADReSS utterance-level features..."
echo ""

# Run different model configurations on your ADReSS data

# 1. Lexical-only model (just your CHAT features)
echo "1. Training lexical-only model..."
echo "   Using your 400-dimensional utterance features from CHAT transcripts"
python train/train.py \
    --experiment_name "lexical_only_adress" \
    --model_type lexical_only \
    --lexical_dim 400 \
    --val_csv $TEST_CSV \
    --test_csv $TEST_CSV \
    --output_dir $OUTPUT_DIR \
    $COMMON_ARGS

echo ""

# 2. ERM training (standard approach)
echo "2. Training with ERM (standard training)..."
python train/train_erm.py \
    --experiment_name "erm_lexical_adress" \
    --model_type lexical_only \
    --lexical_dim 400 \
    --val_csv $TEST_CSV \
    --log_dir $OUTPUT_DIR/logs \
    $COMMON_ARGS

echo ""

# 3. GroupDRO training (robust to group differences)
echo "3. Training with GroupDRO (robust training)..."
python train/train_groupdro.py \
    --experiment_name "groupdro_lexical_adress" \
    --model_type lexical_only \
    --lexical_dim 400 \
    --val_csv $TEST_CSV \
    --log_dir $OUTPUT_DIR/logs \
    $COMMON_ARGS

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Your experiments:"
echo "- lexical_only_adress: Basic lexical model with your CHAT features"
echo "- erm_lexical_adress: Standard ERM training"
echo "- groupdro_lexical_adress: GroupDRO for robust training"
echo ""
echo "Each experiment folder contains:"
echo "- results.json: Final metrics and performance"
echo "- model.pth: Saved model weights"
echo "- training logs"
    fi
    
    echo ""
    
    # Frozen WavLM baseline LOCO  
    echo "Running frozen WavLM LOCO baseline..."
    if python baselines/run_wavlm_loco.py \
        --corpus_config "$CORPUS_CONFIG" \
        --experiment_name "loco_baseline_wavlm" \
        --pooling_strategy attention \
        --classifier_type logistic \
        --output_dir "$OUTPUT_DIR" \
        --seed $SEED 2>&1 | tee "$OUTPUT_DIR/logs/loco_baseline_wavlm.log"; then
        echo "✓ Completed frozen WavLM LOCO baseline"
    else
        echo "✗ Failed frozen WavLM LOCO baseline"
    fi
    
    echo ""
}

# Main execution
echo "Starting LOCO evaluation experiments..."
start_time=$(date +%s)

# Run baseline LOCO experiments first
run_baseline_loco

# Main SALSA LOCO experiments
echo "Running main SALSA LOCO experiments..."
echo ""

# ERM LOCO (no GroupDRO)
if run_loco_experiment "loco_erm_multimodal" "false" \
    "Multimodal SALSA with ERM training - tests generalization without robust optimization"; then
    echo "ERM LOCO experiment completed"
fi

echo ""

# GroupDRO LOCO (with GroupDRO)
if run_loco_experiment "loco_groupdro_multimodal" "true" \
    "Multimodal SALSA with GroupDRO training - tests robust cross-corpus generalization"; then
    echo "GroupDRO LOCO experiment completed"
fi

echo ""

# Acoustic-only LOCO
echo "Running acoustic-only LOCO experiment..."
if python eval/leave_one_corpus.py \
    --corpus_config "$CORPUS_CONFIG" \
    --experiment_name "loco_acoustic_only" \
    --acoustic_dim 768 \
    --lexical_dim 512 \
    --model_type acoustic_only \
    $COMMON_ARGS 2>&1 | tee "$OUTPUT_DIR/logs/loco_acoustic_only.log"; then
    echo "✓ Completed acoustic-only LOCO"
else
    echo "✗ Failed acoustic-only LOCO" 
fi

echo ""

# Lexical-only LOCO
echo "Running lexical-only LOCO experiment..."
if python eval/leave_one_corpus.py \
    --corpus_config "$CORPUS_CONFIG" \
    --experiment_name "loco_lexical_only" \
    --acoustic_dim 768 \
    --lexical_dim 512 \
    --model_type lexical_only \
    $COMMON_ARGS 2>&1 | tee "$OUTPUT_DIR/logs/loco_lexical_only.log"; then
    echo "✓ Completed lexical-only LOCO"
else
    echo "✗ Failed lexical-only LOCO"
fi

echo ""

# LoRA rank ablation LOCO
echo "Running LoRA rank ablation LOCO experiments..."
for lora_rank in 0 4 8 16; do
    experiment_name="loco_lora_rank_${lora_rank}"
    description="LOCO with LoRA rank $lora_rank"
    
    echo "Running LOCO with LoRA rank $lora_rank..."
    if python eval/leave_one_corpus.py \
        --corpus_config "$CORPUS_CONFIG" \
        --experiment_name "$experiment_name" \
        --lora_rank $lora_rank \
        --use_group_dro \
        $COMMON_ARGS 2>&1 | tee "$OUTPUT_DIR/logs/${experiment_name}.log"; then
        echo "✓ Completed LOCO LoRA rank $lora_rank"
    else
        echo "✗ Failed LOCO LoRA rank $lora_rank"
    fi
done

echo ""

# Generate comprehensive LOCO analysis
echo "Generating comprehensive LOCO analysis..."

if python eval/analyze_loco_results.py \
    --results_dir "$OUTPUT_DIR" \
    --output_file "$OUTPUT_DIR/loco_comprehensive_analysis.html" \
    --include_plots 2>&1 | tee "$OUTPUT_DIR/logs/loco_analysis.log"; then
    echo "✓ LOCO analysis completed"
else
    echo "✗ LOCO analysis failed"
fi

# Create comparison plots
echo "Creating LOCO comparison visualizations..."

if python eval/plot_loco_comparisons.py \
    --results_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR/plots" \
    --create_radar_plots \
    --create_heatmaps 2>&1 | tee "$OUTPUT_DIR/logs/loco_plots.log"; then
    echo "✓ LOCO visualization completed"
else
    echo "✗ LOCO visualization failed"
fi

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "=========================================="
echo "LOCO Evaluation Completed!"
echo "=========================================="
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Summary of results
echo "Key results files:"
echo "  - Comprehensive analysis: $OUTPUT_DIR/loco_comprehensive_analysis.html"
echo "  - Individual LOCO results: $OUTPUT_DIR/*_loco_results.json"
echo "  - Summary reports: $OUTPUT_DIR/loco_summary_report.txt"
echo "  - Plots and visualizations: $OUTPUT_DIR/plots/"
echo "  - Logs: $OUTPUT_DIR/logs/"
echo ""

# Print quick summary if analysis was successful
if [[ -f "$OUTPUT_DIR/loco_summary_report.txt" ]]; then
    echo "Quick Summary:"
    echo "=============="
    
    # Extract key metrics from summary report
    if grep -q "Mean Macro F1:" "$OUTPUT_DIR/loco_summary_report.txt"; then
        echo "Cross-corpus generalization results:"
        grep "Mean Macro F1:" "$OUTPUT_DIR/loco_summary_report.txt" | head -3
        echo ""
        grep "Mean Worst Group F1:" "$OUTPUT_DIR/loco_summary_report.txt" | head -1 2>/dev/null || true
    fi
fi

echo ""
echo "Interpretation guide:"
echo "  - Lower performance gaps between within-corpus and LOCO indicate better generalization"
echo "  - GroupDRO should show improved worst-group performance compared to ERM"
echo "  - Multimodal fusion should outperform single-modality approaches"
echo "  - Higher LoRA ranks may improve performance but increase parameters"
echo ""
echo "Next steps:"
echo "  1. Review the comprehensive analysis report"
echo "  2. Compare LOCO results with within-corpus results" 
echo "  3. Run ablation studies: ./scripts/run_ablations.sh"
echo "  4. Consider noise robustness testing"