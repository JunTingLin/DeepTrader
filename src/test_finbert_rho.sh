#!/bin/bash

# Test validation and test with FinBERT-generated rho
# Usage: bash src/test_finbert_rho.sh [experiment_prefix] [mode] [window_days]
# Example: bash src/test_finbert_rho.sh src/outputs/0318/131246 positive_prob 5
#
# This script:
# 1. Backs up original val_results.json and test_results.json (if they exist)
# 2. Reads hyper.json to determine val_idx, test_idx, test_idx_end
# 3. Finds corresponding FinBERT rho files based on index ranges
# 4. Runs validation and test with FinBERT rho

set -e

if [ $# -lt 1 ]; then
    echo "Usage: bash src/test_finbert_rho.sh [experiment_prefix] [mode] [window_days]"
    echo "Example: bash src/test_finbert_rho.sh src/outputs/0318/131246 positive_prob 5"
    echo ""
    echo "Arguments:"
    echo "  experiment_prefix  Path to experiment directory (required)"
    echo "  mode               Rho calculation mode (default: positive_prob)"
    echo "                     Options: positive_prob, sentiment_diff, weighted"
    echo "  window_days        Rolling window size in days (default: 5)"
    exit 1
fi

EXPERIMENT_PREFIX=$1
MODE=${2:-positive_prob}
WINDOW_DAYS=${3:-5}

# Check if experiment directory exists
if [ ! -d "$EXPERIMENT_PREFIX" ]; then
    echo "Error: Experiment directory $EXPERIMENT_PREFIX does not exist"
    exit 1
fi

# Read hyper.json to get indices and data_prefix
HYPER_FILE="$EXPERIMENT_PREFIX/log_file/hyper.json"
if [ ! -f "$HYPER_FILE" ]; then
    echo "Error: hyper.json not found at $HYPER_FILE"
    exit 1
fi

# Extract values from hyper.json using python
VAL_IDX=$(python3 -c "import json; print(json.load(open('$HYPER_FILE'))['val_idx'])")
TEST_IDX=$(python3 -c "import json; print(json.load(open('$HYPER_FILE'))['test_idx'])")
TEST_IDX_END=$(python3 -c "import json; print(json.load(open('$HYPER_FILE'))['test_idx_end'])")
DATA_PREFIX=$(python3 -c "import json; print(json.load(open('$HYPER_FILE'))['data_prefix'])")

echo "============================================"
echo "FinBERT Rho Testing"
echo "============================================"
echo "Experiment: $EXPERIMENT_PREFIX"
echo "Data prefix: $DATA_PREFIX"
echo "Val range: $VAL_IDX - $TEST_IDX"
echo "Test range: $TEST_IDX - $TEST_IDX_END"
echo "Mode: $MODE"
echo "Window days: $WINDOW_DAYS"
echo ""

# Backup original results if they exist (only once, don't overwrite existing backups)
JSON_DIR="$EXPERIMENT_PREFIX/json_file"

if [ -f "$JSON_DIR/val_results.json" ] && [ ! -f "$JSON_DIR/val_results_origin.json" ]; then
    cp "$JSON_DIR/val_results.json" "$JSON_DIR/val_results_origin.json"
    echo "✅ Backed up val_results.json -> val_results_origin.json"
fi

if [ -f "$JSON_DIR/test_results.json" ] && [ ! -f "$JSON_DIR/test_results_origin.json" ]; then
    cp "$JSON_DIR/test_results.json" "$JSON_DIR/test_results_origin.json"
    echo "✅ Backed up test_results.json -> test_results_origin.json"
fi

# Construct FinBERT rho file paths based on index ranges
FINBERT_RHO_VAL_FILE="${DATA_PREFIX}finbert_rho_${VAL_IDX}-${TEST_IDX}_mode-${MODE}_window-${WINDOW_DAYS}.json"
FINBERT_RHO_TEST_FILE="${DATA_PREFIX}finbert_rho_${TEST_IDX}-${TEST_IDX_END}_mode-${MODE}_window-${WINDOW_DAYS}.json"

echo ""
echo "FinBERT rho files:"
echo "  Val:  $FINBERT_RHO_VAL_FILE"
echo "  Test: $FINBERT_RHO_TEST_FILE"
echo ""

# Check if FinBERT rho files exist
if [ ! -f "$FINBERT_RHO_VAL_FILE" ]; then
    echo "Error: FinBERT rho val file not found: $FINBERT_RHO_VAL_FILE"
    echo ""
    echo "Please generate FinBERT rho files first:"
    echo "  python src/data/generate_finbert_rho.py --config ${DATA_PREFIX}finbert_rho_config.json"
    exit 1
fi

if [ ! -f "$FINBERT_RHO_TEST_FILE" ]; then
    echo "Error: FinBERT rho test file not found: $FINBERT_RHO_TEST_FILE"
    echo ""
    echo "Please generate FinBERT rho files first:"
    echo "  python src/data/generate_finbert_rho.py --config ${DATA_PREFIX}finbert_rho_config.json"
    exit 1
fi

# Output file names with mode and window
VAL_OUTPUT="val_results_finbert_rho_mode-${MODE}_window-${WINDOW_DAYS}.json"
TEST_OUTPUT="test_results_finbert_rho_mode-${MODE}_window-${WINDOW_DAYS}.json"

# Run validation with FinBERT rho
echo "-------------------------------------------"
echo "Running Validation with FinBERT rho..."
echo "-------------------------------------------"
python src/validate.py --prefix "$EXPERIMENT_PREFIX" --ground_truth_rho "$FINBERT_RHO_VAL_FILE"

if [ -f "$JSON_DIR/val_results.json" ]; then
    mv "$JSON_DIR/val_results.json" "$JSON_DIR/$VAL_OUTPUT"
    echo "✅ Results saved to: $VAL_OUTPUT"
fi

# Run test with FinBERT rho
echo ""
echo "-------------------------------------------"
echo "Running Test with FinBERT rho..."
echo "-------------------------------------------"
python src/test.py --prefix "$EXPERIMENT_PREFIX" --ground_truth_rho "$FINBERT_RHO_TEST_FILE"

if [ -f "$JSON_DIR/test_results.json" ]; then
    mv "$JSON_DIR/test_results.json" "$JSON_DIR/$TEST_OUTPUT"
    echo "✅ Results saved to: $TEST_OUTPUT"
fi

echo ""
echo "============================================"
echo "✅ FinBERT rho testing complete!"
echo ""
echo "Results saved to:"
echo "  - $JSON_DIR/$VAL_OUTPUT"
echo "  - $JSON_DIR/$TEST_OUTPUT"
if [ -f "$JSON_DIR/val_results_origin.json" ]; then
    echo ""
    echo "Original results backed up to:"
    echo "  - $JSON_DIR/val_results_origin.json"
    echo "  - $JSON_DIR/test_results_origin.json"
fi
echo "============================================"
