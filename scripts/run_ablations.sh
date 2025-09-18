#!/bin/bash

# SALSA Ablation Study Runner
# Comprehensive ablation studies across architectural choices, training strategies, and hyperparameters

set -e  # Exit on any error

# Default parameters
DATA_DIR="./data"
OUTPUT_DIR="./ablation_results"
DEVICE="auto"
BATCH_SIZE=16
MAX_EPOCHS=30
PATIENCE=8
SEED=42
DATASET="adress"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
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
        --quick)
            # Quick mode: reduced epochs and simpler ablations
            MAX_EPOCHS=15
            PATIENCE=5
            echo "Quick mode enabled: MAX_EPOCHS=$MAX_EPOCHS, PATIENCE=$PATIENCE"
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data-dir DIR        Data directory (default: ./data)"
            echo "  --output-dir DIR      Output directory (default: ./ablation_results)"
            echo "  --dataset DATASET     Dataset to use: adress, adresso, pitt (default: adress)"
            echo "  --device DEVICE       Device to use: cuda, cpu, or auto (default: auto)"
            echo "  --batch-size SIZE     Batch size (default: 16)"
            echo "  --max-epochs N        Maximum epochs (default: 30)"
            echo "  --patience N          Early stopping patience (default: 8)"
            echo "  --seed N              Random seed (default: 42)"
            echo "  --quick              Quick mode with reduced epochs"
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
echo "SALSA Comprehensive Ablation Studies"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Dataset: $DATASET"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "Max epochs: $MAX_EPOCHS"
echo "Patience: $PATIENCE"
echo "Random seed: $SEED"
echo ""

# Validate dataset choice
case "$DATASET" in
    adress|adresso|pitt)
        echo "Using dataset: $DATASET"
        ;;
    *)
        echo "ERROR: Invalid dataset '$DATASET'. Choose from: adress, adresso, pitt"
        exit 1
        ;;
esac

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/results"
mkdir -p "$OUTPUT_DIR/plots"
mkdir -p "$OUTPUT_DIR/configs"

# Set dataset-specific paths
DATA_PATH="$DATA_DIR/$DATASET"
TRAIN_MANIFEST="$DATA_PATH/train_manifest.json"
VAL_MANIFEST="$DATA_PATH/val_manifest.json"
TEST_MANIFEST="$DATA_PATH/test_manifest.json"

# Validate dataset files
echo "Validating dataset files..."
for manifest in "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$TEST_MANIFEST"; do
    if [[ ! -f "$manifest" ]]; then
        echo "ERROR: Missing manifest file: $manifest"
        echo "Please run data preparation scripts first."
        exit 1
    fi
done
echo "✓ All dataset files found"

# Common arguments for all experiments  
COMMON_ARGS="--train_manifest $TRAIN_MANIFEST --val_manifest $VAL_MANIFEST --test_manifest $TEST_MANIFEST --batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS --patience $PATIENCE --seed $SEED"

if [ "$DEVICE" != "auto" ]; then
    COMMON_ARGS="$COMMON_ARGS --device $DEVICE"
fi

# Function to run ablation experiment
run_ablation() {
    local experiment_name=$1
    local description="$2"
    shift 2  # Remove first two arguments
    local extra_args="$*"  # All remaining arguments
    
    echo "----------------------------------------"
    echo "Running Ablation: $experiment_name"
    echo "Description: $description"
    echo "Extra Args: $extra_args"
    echo "----------------------------------------"
    
    local log_file="$OUTPUT_DIR/logs/${experiment_name}.log"
    local results_file="$OUTPUT_DIR/results/${experiment_name}.json"
    
    echo "Ablation Study: $experiment_name" > "$log_file"
    echo "Started at: $(date)" >> "$log_file"
    echo "Description: $description" >> "$log_file"
    echo "Args: $extra_args" >> "$log_file"
    echo "" >> "$log_file"
    
    if python train/train.py \
        --experiment_name "$experiment_name" \
        --output_dir "$OUTPUT_DIR/results" \
        $COMMON_ARGS \
        $extra_args 2>&1 | tee -a "$log_file"; then
        
        echo "✓ Completed ablation: $experiment_name"
        echo "Finished at: $(date)" >> "$log_file"
        return 0
    else
        echo "✗ Failed ablation: $experiment_name"  
        echo "FAILED at: $(date)" >> "$log_file"
        return 1
    fi
}

# Initialize results tracking
echo "experiment_name,description,completed,macro_f1,auroc,auprc" > "$OUTPUT_DIR/ablation_summary.csv"

# Function to log experiment result
log_result() {
    local experiment_name=$1
    local description="$2"
    local completed=$3
    local results_file="$OUTPUT_DIR/results/${experiment_name}.json"
    
    local macro_f1="N/A"
    local auroc="N/A"
    local auprc="N/A"
    
    if [[ "$completed" == "true" && -f "$results_file" ]]; then
        # Extract metrics from results file
        macro_f1=$(python -c "import json; data=json.load(open('$results_file')); print(f\"{data.get('test_metrics', {}).get('macro_f1', 'N/A'):.4f}\" if isinstance(data.get('test_metrics', {}).get('macro_f1'), (int, float)) else 'N/A')" 2>/dev/null || echo "N/A")
        auroc=$(python -c "import json; data=json.load(open('$results_file')); print(f\"{data.get('test_metrics', {}).get('auroc', 'N/A'):.4f}\" if isinstance(data.get('test_metrics', {}).get('auroc'), (int, float)) else 'N/A')" 2>/dev/null || echo "N/A")
        auprc=$(python -c "import json; data=json.load(open('$results_file')); print(f\"{data.get('test_metrics', {}).get('auprc', 'N/A'):.4f}\" if isinstance(data.get('test_metrics', {}).get('auprc'), (int, float)) else 'N/A')" 2>/dev/null || echo "N/A")
    fi
    
    echo "$experiment_name,\"$description\",$completed,$macro_f1,$auroc,$auprc" >> "$OUTPUT_DIR/ablation_summary.csv"
}

start_time=$(date +%s)

echo "Starting comprehensive ablation studies..."
echo ""

# ============================================================================
# 1. CORE ARCHITECTURAL ABLATIONS
# ============================================================================

echo "1. CORE ARCHITECTURAL ABLATIONS"
echo "================================="

# Multimodal vs. Unimodal
if run_ablation "arch_multimodal_full" \
    "Full multimodal SALSA with gated cross-attention fusion" \
    --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512 --use_cross_attention true --dropout 0.1; then
    log_result "arch_multimodal_full" "Full multimodal SALSA with gated cross-attention fusion" "true"
else
    log_result "arch_multimodal_full" "Full multimodal SALSA with gated cross-attention fusion" "false"
fi

if run_ablation "arch_acoustic_only" \
    "Acoustic branch only (WavLM + LoRA)" \
    --model_type acoustic_only --acoustic_dim 768 --dropout 0.1; then
    log_result "arch_acoustic_only" "Acoustic branch only (WavLM + LoRA)" "true"
else
    log_result "arch_acoustic_only" "Acoustic branch only (WavLM + LoRA)" "false"
fi

if run_ablation "arch_lexical_only" \
    "Lexical branch only (Whisper + MiniLM + linguistic features)" \
    --model_type lexical_only --lexical_dim 512 --dropout 0.1; then
    log_result "arch_lexical_only" "Lexical branch only (Whisper + MiniLM + linguistic features)" "true"
else
    log_result "arch_lexical_only" "Lexical branch only (Whisper + MiniLM + linguistic features)" "false"
fi

echo ""

# ============================================================================
# 2. FUSION STRATEGY ABLATIONS
# ============================================================================

echo "2. FUSION STRATEGY ABLATIONS"  
echo "============================="

# Cross-attention vs. simple concatenation
if run_ablation "fusion_cross_attention" \
    "Multimodal fusion with gated cross-attention" \
    --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512 --use_cross_attention true --dropout 0.1; then
    log_result "fusion_cross_attention" "Multimodal fusion with gated cross-attention" "true"
else
    log_result "fusion_cross_attention" "Multimodal fusion with gated cross-attention" "false"
fi

if run_ablation "fusion_concatenation" \
    "Simple concatenation fusion" \
    --acoustic_dim 768 --lexical_dim 512 --fusion_dim 1280 --use_cross_attention false --dropout 0.1; then
    log_result "fusion_concatenation" "Simple concatenation fusion" "true"
else
    log_result "fusion_concatenation" "Simple concatenation fusion" "false"
fi

if run_ablation "fusion_addition" \
    "Element-wise addition fusion (requires same dims)" \
    --acoustic_dim 512 --lexical_dim 512 --fusion_dim 512 --fusion_type addition --dropout 0.1; then
    log_result "fusion_addition" "Element-wise addition fusion" "true"
else  
    log_result "fusion_addition" "Element-wise addition fusion" "false"
fi

if run_ablation "fusion_attention_pooling" \
    "Attention-based pooling fusion" \
    --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512 --fusion_type attention --dropout 0.1; then
    log_result "fusion_attention_pooling" "Attention-based pooling fusion" "true"
else
    log_result "fusion_attention_pooling" "Attention-based pooling fusion" "false"
fi

echo ""

# ============================================================================  
# 3. LORA RANK ABLATIONS
# ============================================================================

echo "3. LORA RANK ABLATIONS"
echo "======================="

for lora_rank in 0 4 8 16 32; do
    experiment_name="lora_rank_${lora_rank}"
    description="LoRA rank $lora_rank (0=frozen WavLM, higher=more parameters)"
    
    if run_ablation "$experiment_name" \
        "$description" \
        --lora_rank $lora_rank --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512 --dropout 0.1; then
        log_result "$experiment_name" "$description" "true"
    else
        log_result "$experiment_name" "$description" "false"
    fi
done

echo ""

# ============================================================================
# 4. TRAINING STRATEGY ABLATIONS
# ============================================================================

echo "4. TRAINING STRATEGY ABLATIONS"
echo "==============================="

# ERM vs. GroupDRO
if run_ablation "train_erm" \
    "Empirical Risk Minimization (standard training)" \
    --use_group_dro false --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512 --dropout 0.1; then
    log_result "train_erm" "Empirical Risk Minimization (standard training)" "true"
else
    log_result "train_erm" "Empirical Risk Minimization (standard training)" "false"  
fi

if run_ablation "train_group_dro" \
    "Group Distributionally Robust Optimization" \
    --use_group_dro true --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512 --dropout 0.1; then
    log_result "train_group_dro" "Group Distributionally Robust Optimization" "true"
else
    log_result "train_group_dro" "Group Distributionally Robust Optimization" "false"
fi

# Loss function ablations
if run_ablation "loss_focal" \
    "Focal loss for class imbalance" \
    --loss_type focal --focal_alpha 0.25 --focal_gamma 2.0 --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512 --dropout 0.1; then
    log_result "loss_focal" "Focal loss for class imbalance" "true"
else
    log_result "loss_focal" "Focal loss for class imbalance" "false"
fi

if run_ablation "loss_cross_entropy" \
    "Standard cross-entropy loss" \
    --loss_type cross_entropy --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512 --dropout 0.1; then
    log_result "loss_cross_entropy" "Standard cross-entropy loss" "true"
else
    log_result "loss_cross_entropy" "Standard cross-entropy loss" "false"
fi

if run_ablation "loss_label_smoothing" \
    "Cross-entropy with label smoothing" \
    --loss_type cross_entropy --label_smoothing 0.1 --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512 --dropout 0.1; then
    log_result "loss_label_smoothing" "Cross-entropy with label smoothing" "true"
else
    log_result "loss_label_smoothing" "Cross-entropy with label smoothing" "false"
fi

echo ""

# ============================================================================
# 5. HYPERPARAMETER ABLATIONS  
# ============================================================================

echo "5. HYPERPARAMETER ABLATIONS"
echo "============================"

# Learning rate ablation
for lr in 1e-5 5e-5 1e-4 5e-4 1e-3; do
    experiment_name="lr_${lr}"
    description="Learning rate $lr"
    
    if run_ablation "$experiment_name" \
        "$description" \
        --learning_rate $lr --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512 --dropout 0.1; then
        log_result "$experiment_name" "$description" "true"
    else
        log_result "$experiment_name" "$description" "false"
    fi
done

# Dropout ablation
for dropout in 0.0 0.1 0.2 0.3 0.5; do
    experiment_name="dropout_${dropout}"
    description="Dropout rate $dropout"
    
    if run_ablation "$experiment_name" \
        "$description" \
        --dropout $dropout --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512; then
        log_result "$experiment_name" "$description" "true"
    else
        log_result "$experiment_name" "$description" "false"
    fi
done

# Weight decay ablation
for wd in 0.0 1e-6 1e-5 1e-4 1e-3; do
    experiment_name="wd_${wd}"
    description="Weight decay $wd"
    
    if run_ablation "$experiment_name" \
        "$description" \
        --weight_decay $wd --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512 --dropout 0.1; then
        log_result "$experiment_name" "$description" "true"
    else
        log_result "$experiment_name" "$description" "false"
    fi
done

echo ""

# ============================================================================
# 6. ARCHITECTURAL DEPTH ABLATIONS
# ============================================================================

echo "6. ARCHITECTURAL DEPTH ABLATIONS"
echo "================================="

# Number of fusion layers
for n_layers in 1 2 3 4; do
    experiment_name="fusion_layers_${n_layers}"
    description="Number of fusion layers: $n_layers"
    
    if run_ablation "$experiment_name" \
        "$description" \
        --n_fusion_layers $n_layers --acoustic_dim 768 --lexical_dim 512 --fusion_dim 512 --dropout 0.1; then
        log_result "$experiment_name" "$description" "true"
    else
        log_result "$experiment_name" "$description" "false"
    fi
done

# Hidden dimensions ablation  
for hidden_dim in 256 512 768 1024; do
    experiment_name="hidden_dim_${hidden_dim}"
    description="Hidden dimension: $hidden_dim"
    
    if run_ablation "$experiment_name" \
        "$description" \
        --fusion_dim $hidden_dim --acoustic_dim 768 --lexical_dim 512 --dropout 0.1; then
        log_result "$experiment_name" "$description" "true"
    else
        log_result "$experiment_name" "$description" "false"
    fi
done

echo ""

# ============================================================================
# 7. GENERATE ANALYSIS AND VISUALIZATIONS
# ============================================================================

echo "7. ANALYSIS AND VISUALIZATION"
echo "=============================="

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "Generating ablation analysis report..."

if python eval/analyze_ablations.py \
    --results_dir "$OUTPUT_DIR/results" \
    --summary_file "$OUTPUT_DIR/ablation_summary.csv" \
    --output_file "$OUTPUT_DIR/ablation_analysis.html" \
    --dataset "$DATASET" 2>&1 | tee "$OUTPUT_DIR/logs/ablation_analysis.log"; then
    echo "✓ Ablation analysis completed"
else
    echo "✗ Ablation analysis failed"
fi

echo "Creating ablation visualizations..."

if python eval/plot_ablation_results.py \
    --results_dir "$OUTPUT_DIR/results" \
    --summary_file "$OUTPUT_DIR/ablation_summary.csv" \
    --output_dir "$OUTPUT_DIR/plots" \
    --create_comparison_plots \
    --create_sensitivity_analysis 2>&1 | tee "$OUTPUT_DIR/logs/ablation_plots.log"; then
    echo "✓ Ablation visualizations completed"
else
    echo "✗ Ablation visualizations failed"
fi

echo ""
echo "=========================================="
echo "Ablation Studies Completed!"
echo "=========================================="
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo "Results saved to: $OUTPUT_DIR"
echo ""

echo "Key results files:"
echo "  - Summary table: $OUTPUT_DIR/ablation_summary.csv"
echo "  - Detailed analysis: $OUTPUT_DIR/ablation_analysis.html"
echo "  - Individual results: $OUTPUT_DIR/results/*.json"
echo "  - Plots and charts: $OUTPUT_DIR/plots/"
echo "  - Logs: $OUTPUT_DIR/logs/"
echo ""

# Print quick summary
if [[ -f "$OUTPUT_DIR/ablation_summary.csv" ]]; then
    echo "Quick Summary (Top 5 by Macro F1):"
    echo "==================================="
    
    # Sort by macro_f1 and show top 5 (skip header, sort by 4th column numerically, reverse order)
    {
        head -1 "$OUTPUT_DIR/ablation_summary.csv"
        tail -n +2 "$OUTPUT_DIR/ablation_summary.csv" | grep ",true," | sort -t, -k4 -nr | head -5
    } | column -t -s,
    
    echo ""
fi

echo "Key findings to look for:"
echo "  - Architecture: Does multimodal outperform unimodal?"
echo "  - Fusion: Which fusion strategy works best?"
echo "  - LoRA: What's the optimal rank vs. parameter trade-off?"
echo "  - Training: Does GroupDRO improve worst-case performance?"
echo "  - Loss: Which loss function handles class imbalance best?"
echo "  - Hyperparams: What are the optimal learning rate, dropout, weight decay?"
echo ""
echo "Next steps:"
echo "  1. Review the detailed analysis report"
echo "  2. Check sensitivity plots for hyperparameter robustness"
echo "  3. Compare with LOCO results for generalization insights"
echo "  4. Consider ensemble methods based on top performers"