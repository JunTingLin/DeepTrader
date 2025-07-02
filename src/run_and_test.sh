#!/bin/bash

# Automated script to run training (run.py) followed by testing (test.py)
# Usage:
#   ./run_and_test.sh -c hyper.json    # Use specified config file (required)

set -e  # Exit on any error

echo "========================================"
echo "DeepTrader Automated Training & Testing"
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
    echo "Example: $0 -c hyper.json"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file does not exist: $CONFIG"
    exit 1
fi

echo "Using config file: $CONFIG"
echo "Starting training..."

# Build run.py command
RUN_CMD="python run.py -c $CONFIG"

echo "Executing command: $RUN_CMD"

# Execute training and capture output to extract PREFIX
OUTPUT_FILE=$(mktemp)
if eval "$RUN_CMD" 2>&1 | tee "$OUTPUT_FILE"; then
    echo "Training completed!"
    
    # Extract PREFIX from output (pattern: outputs/MMDD/HHMMSS)
    PREFIX=$(grep -o "outputs/[0-9]\{4\}/[0-9]\{6\}" "$OUTPUT_FILE" | tail -1)
    
    # If not found, try to find the latest directory
    if [ -z "$PREFIX" ]; then
        if [ -d "outputs" ]; then
            PREFIX=$(find outputs -type d -name "[0-9][0-9][0-9][0-9][0-9][0-9]" | sort | tail -1)
            if [ -n "$PREFIX" ]; then
                PREFIX=$(dirname "$PREFIX")/$(basename "$PREFIX")
            fi
        fi
    fi
    
    rm -f "$OUTPUT_FILE"
    
    if [ -z "$PREFIX" ]; then
        echo "Error: Unable to determine output directory PREFIX"
        exit 1
    fi
    
    echo "Generated PREFIX: $PREFIX"
else
    rm -f "$OUTPUT_FILE"
    echo "Training failed!"
    exit 1
fi

# Check if PREFIX directory exists
if [ ! -d "$PREFIX" ]; then
    echo "Error: PREFIX directory does not exist: $PREFIX"
    exit 1
fi

# Check if model directory exists
if [ ! -d "$PREFIX/model_file" ]; then
    echo "Error: Model directory does not exist: $PREFIX/model_file"
    exit 1
fi

echo ""
echo "========================================"
echo "Starting testing..."
echo "========================================"

# Build test.py command
TEST_CMD="python test.py --prefix \"$PREFIX\""

echo "Executing command: $TEST_CMD"

# Execute testing
if eval "$TEST_CMD"; then
    echo ""
    echo "========================================"
    echo "All completed successfully!"
    echo "Results saved in: $PREFIX"
    echo "========================================"
else
    echo "Testing failed!"
    exit 1
fi