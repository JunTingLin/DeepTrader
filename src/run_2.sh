#!/bin/bash

# Automated script to run sliding window training (run_2.py)
# Usage (from project root):
#   bash src/run_2.sh -c src/hyper_sliding.json

set -e  # Exit on any error

echo "========================================"
echo "DeepTrader Sliding Window Training"
echo "========================================"

CONFIG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 -c CONFIG"
            exit 1
            ;;
    esac
done

# Check if config is provided
if [ -z "$CONFIG" ]; then
    echo "Error: Config file is required!"
    echo "Usage: $0 -c CONFIG"
    echo "Example: $0 -c src/hyper_sliding.json"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file does not exist: $CONFIG"
    exit 1
fi

echo "Using config file: $CONFIG"
echo "Starting sliding window training..."

# Build run_2.py command
RUN_CMD="python src/run_2.py -c $CONFIG"

echo "Executing command: $RUN_CMD"

# Execute training and capture output to extract PREFIX
OUTPUT_FILE=$(mktemp)
if eval "$RUN_CMD" 2>&1 | tee "$OUTPUT_FILE"; then
    echo ""
    echo "========================================"
    echo "Sliding window training completed!"

    # Extract PREFIX from run_2.py output (looking for [DEEPTRADER_PREFIX] marker)
    PREFIX=$(grep "\[DEEPTRADER_PREFIX\]" "$OUTPUT_FILE" | cut -d' ' -f2)

    rm -f "$OUTPUT_FILE"

    if [ -n "$PREFIX" ]; then
        echo "Results saved in: $PREFIX"
    fi
    echo "========================================"
else
    rm -f "$OUTPUT_FILE"
    echo "Training failed!"
    exit 1
fi
