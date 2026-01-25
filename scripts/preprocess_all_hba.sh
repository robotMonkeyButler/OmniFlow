#!/bin/bash
# Batch preprocess all HBA sub-datasets with GloVe text features

# Configuration
GLOVE_PATH="${GLOVE_PATH:-glove.6B.300d.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-preprocessed_data}"
TEXT_POLICY="${TEXT_POLICY:-problem}"
FEATURE_ROOT="${FEATURE_ROOT:-}"

# Check if GloVe file exists
if [ ! -f "$GLOVE_PATH" ]; then
    echo "Error: GloVe file not found at $GLOVE_PATH"
    echo "Please set GLOVE_PATH environment variable or place glove.6B.300d.txt in current directory"
    exit 1
fi

if [ -z "$FEATURE_ROOT" ]; then
    echo "Error: FEATURE_ROOT is not set."
    echo "Please set FEATURE_ROOT to the directory containing pose/ and opensmile/ subfolders."
    exit 1
fi

# List of all supported datasets
DATASETS=(
    "cremad"
    "iemocap"
    "meld_emotion"
    "meld_senti"
    "chsimsv2"
    "ptsd_in_the_wild"
    "daicwoz"
    "tess"
)

echo "================================================"
echo "HBA Dataset Preprocessing Pipeline"
echo "================================================"
echo "GloVe path: $GLOVE_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Text policy: $TEXT_POLICY"
echo "Datasets: ${DATASETS[@]}"
echo "Feature root: $FEATURE_ROOT"
echo "================================================"
echo ""

# Process each dataset
FAILED_DATASETS=()
SUCCESS_COUNT=0

for dataset in "${DATASETS[@]}"; do
    echo "▶ Processing: $dataset"
    echo "----------------------------------------"
    
    if python scripts/preprocess_hba.py \
        --dataset_id "$dataset" \
        --glove_path "$GLOVE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --feature_root "$FEATURE_ROOT" \
        --text_policy "$TEXT_POLICY"; then
        
        echo "✓ $dataset completed successfully"
        ((SUCCESS_COUNT++))
    else
        echo "✗ $dataset failed"
        FAILED_DATASETS+=("$dataset")
    fi
    
    echo ""
done

# Summary
echo "================================================"
echo "Preprocessing Summary"
echo "================================================"
echo "Total datasets: ${#DATASETS[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: ${#FAILED_DATASETS[@]}"

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo ""
    echo "Failed datasets:"
    for dataset in "${FAILED_DATASETS[@]}"; do
        echo "  - $dataset"
    done
    exit 1
else
    echo ""
    echo "All datasets preprocessed successfully! 🎉"
    echo "Output directory: $OUTPUT_DIR"
fi
