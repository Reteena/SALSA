#!/bin/bash

# SALSA Leave-One-Corpus-Out (LOCO) Evaluation Runner
# Tests cross-corpus generalization by training on N-1 corpora and testing on the held-out corpus

set -e  # Exit on any error

# Default parameters
DATA_DIR="./data"
OUTPUT_DIR="./loco_results"
DEVICE="auto"
BATCH_SIZE=16
MAX_EPOCHS=50
PATIENCE=10
SEED=42

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
            echo "  --data-dir DIR        Data directory (default: ./data)"
            echo "  --output-dir DIR      Output directory (default: ./loco_results)"
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
echo "SALSA Leave-One-Corpus-Out (LOCO) Evaluation"
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
mkdir -p "$OUTPUT_DIR/plots"
mkdir -p "$OUTPUT_DIR/configs"

# Common arguments for all experiments
COMMON_ARGS="--batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS --patience $PATIENCE --seed $SEED --output_dir $OUTPUT_DIR"

if [ "$DEVICE" != "auto" ]; then
    COMMON_ARGS="$COMMON_ARGS --device $DEVICE"
fi

# Create corpus configuration file
CORPUS_CONFIG="$OUTPUT_DIR/configs/corpus_config.json"

echo "Creating corpus configuration..."

cat > "$CORPUS_CONFIG" << EOF
{
  "adress": {
    "name": "ADReSS-2020",
    "data_dir": "$DATA_DIR/adress",
    "train_manifest": "$DATA_DIR/adress/train_manifest.json",
    "val_manifest": "$DATA_DIR/adress/val_manifest.json", 
    "test_manifest": "$DATA_DIR/adress/test_manifest.json",
    "group_id": 0,
    "description": "ADReSS 2020 challenge dataset - balanced picture description task"
  },
  "adresso": {
    "name": "ADReSSo-2021-2022", 
    "data_dir": "$DATA_DIR/adresso",
    "train_manifest": "$DATA_DIR/adresso/train_manifest.json",
    "val_manifest": "$DATA_DIR/adresso/val_manifest.json",
    "test_manifest": "$DATA_DIR/adresso/test_manifest.json", 
    "group_id": 1,
    "description": "ADReSSo 2021/2022 speech-only challenge - more challenging than ADReSS"
  },
  "pitt": {
    "name": "DementiaBank-Pitt",
    "data_dir": "$DATA_DIR/pitt", 
    "train_manifest": "$DATA_DIR/pitt/train_manifest.json",
    "val_manifest": "$DATA_DIR/pitt/val_manifest.json",
    "test_manifest": "$DATA_DIR/pitt/test_manifest.json",
    "group_id": 2,
    "description": "DementiaBank Pittsburgh Cookie Theft corpus - large longitudinal dataset"
  }
}
EOF

echo "Corpus configuration created: $CORPUS_CONFIG"

# Validate corpus configuration
echo "Validating corpus configuration..."

missing_files=()

while IFS= read -r line; do
    if [[ $line =~ \"([^\"]+_manifest\.json)\" ]]; then
        manifest_file="${BASH_REMATCH[1]}"
        if [[ ! -f "$manifest_file" ]]; then
            missing_files+=("$manifest_file")
        fi
    fi
done < "$CORPUS_CONFIG"

if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo "ERROR: Missing manifest files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please ensure all datasets are properly prepared with manifest files."
    echo "Use the data preparation scripts in the preprocess/ directory."
    exit 1
fi

echo "✓ All manifest files found"
echo ""

# Function to run LOCO experiment
run_loco_experiment() {
    local experiment_name=$1
    local use_group_dro=$2
    local description="$3"
    
    echo "----------------------------------------"
    echo "Running LOCO Experiment: $experiment_name"
    echo "Description: $description"
    echo "GroupDRO: $use_group_dro"
    echo "----------------------------------------"
    
    local log_file="$OUTPUT_DIR/logs/${experiment_name}.log"
    echo "LOCO Experiment: $experiment_name" > "$log_file"
    echo "Started at: $(date)" >> "$log_file"
    echo "GroupDRO: $use_group_dro" >> "$log_file"
    echo "" >> "$log_file"
    
    local group_dro_flag=""
    if [[ "$use_group_dro" == "true" ]]; then
        group_dro_flag="--use_group_dro"
    fi
    
    if python eval/leave_one_corpus.py \
        --corpus_config "$CORPUS_CONFIG" \
        --experiment_name "$experiment_name" \
        $group_dro_flag \
        $COMMON_ARGS 2>&1 | tee -a "$log_file"; then
        
        echo "✓ Completed LOCO experiment: $experiment_name"
        echo "Finished at: $(date)" >> "$log_file"
        return 0
    else
        echo "✗ Failed LOCO experiment: $experiment_name"
        echo "FAILED at: $(date)" >> "$log_file"
        return 1
    fi
}

# Function to run baseline LOCO experiments
run_baseline_loco() {
    echo "Running baseline LOCO experiments..."
    echo ""
    
    # openSMILE eGeMAPS baseline LOCO
    echo "Running openSMILE eGeMAPS LOCO baseline..."
    if python baselines/run_opensmile_loco.py \
        --corpus_config "$CORPUS_CONFIG" \
        --experiment_name "loco_baseline_egemaps" \
        --feature_set egemaps \
        --classifier_type logistic \
        --output_dir "$OUTPUT_DIR" \
        --seed $SEED 2>&1 | tee "$OUTPUT_DIR/logs/loco_baseline_egemaps.log"; then
        echo "✓ Completed openSMILE eGeMAPS LOCO baseline"
    else
        echo "✗ Failed openSMILE eGeMAPS LOCO baseline"
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