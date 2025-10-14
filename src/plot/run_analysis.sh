#!/bin/bash

# Simple analysis runner with logging
# Usage: bash run_analysis.sh

# Generate timestamp and log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="analysis_${TIMESTAMP}.log"

echo "Starting analysis at $(date)"
echo "Log will be saved to: $LOG_FILE"

# Activate conda environment and run analysis with logging
source ~/miniconda3/bin/activate DeepTrader-pip && \
python main.py > "$LOG_FILE" 2>&1

echo "Analysis completed at $(date)"
echo "📁 Results saved to: $LOG_FILE"