#!/bin/bash

# Test with ground truth rho values
# Usage: bash test_with_ground_truth_rho.sh [experiment_prefix] [val_gt_file] [test_gt_file]
# Example: bash test_with_ground_truth_rho.sh outputs/1114/035244
# Example: bash test_with_ground_truth_rho.sh outputs/1114/035244 data/fake/val_ground_truth.json data/fake/test_ground_truth.json

if [ $# -eq 0 ]; then
    echo "Usage: bash test_with_ground_truth_rho.sh [experiment_prefix] [val_gt_file] [test_gt_file]"
    echo "Example: bash test_with_ground_truth_rho.sh outputs/1114/035244"
    echo "Example: bash test_with_ground_truth_rho.sh outputs/1114/035244 data/fake/val_ground_truth.json data/fake/test_ground_truth.json"
    exit 1
fi

EXPERIMENT_PREFIX=$1

# Default ground truth file paths (can be overridden by arguments)
VAL_GT_FILE="${2:-data/fake/val_ground_truth.json}"
TEST_GT_FILE="${3:-data/fake/test_ground_truth.json}"

echo "Testing with Ground Truth Rho Values"
echo "Experiment: $EXPERIMENT_PREFIX"
echo "============================================"

# Check if experiment directory exists
if [ ! -d "$EXPERIMENT_PREFIX" ]; then
    echo "Error: Experiment directory $EXPERIMENT_PREFIX does not exist"
    exit 1
fi

# Function to run validation with ground truth rho
run_validation_ground_truth() {
    echo ""
    echo "📊 Running Validation with Ground Truth Rho..."
    echo "-----------------------------------------------"

    GT_FILE="data/fake/val_ground_truth.json"

    if [ ! -f "$GT_FILE" ]; then
        echo "✗ Error: Ground truth file not found: $GT_FILE"
        return 1
    fi

    echo "Using ground truth file: $GT_FILE"
    python validate.py --prefix "$EXPERIMENT_PREFIX" --ground_truth_rho "$GT_FILE"

    if [ -f "$EXPERIMENT_PREFIX/json_file/val_results.json" ]; then
        mv "$EXPERIMENT_PREFIX/json_file/val_results.json" "$EXPERIMENT_PREFIX/json_file/val_results_ground_truth_rho.json"
        echo "✅ Results saved to: val_results_ground_truth_rho.json"
    else
        echo "✗ Error: Validation results not found"
        return 1
    fi
}

# Function to run test with ground truth rho
run_test_ground_truth() {
    echo ""
    echo "🎯 Running Test with Ground Truth Rho..."
    echo "---------------------------------------"

    if [ ! -f "$TEST_GT_FILE" ]; then
        echo "✗ Error: Ground truth file not found: $TEST_GT_FILE"
        return 1
    fi

    echo "Using ground truth file: $TEST_GT_FILE"
    python test.py --prefix "$EXPERIMENT_PREFIX" --ground_truth_rho "$TEST_GT_FILE"

    if [ -f "$EXPERIMENT_PREFIX/json_file/test_results.json" ]; then
        mv "$EXPERIMENT_PREFIX/json_file/test_results.json" "$EXPERIMENT_PREFIX/json_file/test_results_ground_truth_rho.json"
        echo "✅ Results saved to: test_results_ground_truth_rho.json"
    else
        echo "✗ Error: Test results not found"
        return 1
    fi
}

# Run both validation and test with ground truth
run_validation_ground_truth
VAL_STATUS=$?

run_test_ground_truth
TEST_STATUS=$?

echo ""
echo "============================================"

if [ $VAL_STATUS -eq 0 ] && [ $TEST_STATUS -eq 0 ]; then
    echo "🎉 All tests completed successfully!"
    echo ""
    echo "📁 Results available in $EXPERIMENT_PREFIX/json_file/:"
    echo "  - val_results_ground_truth_rho.json (Validation with perfect rho prediction)"
    echo "  - test_results_ground_truth_rho.json (Test with perfect rho prediction)"
    echo ""
    echo "💡 These results show the upper bound of performance with perfect market timing!"
else
    echo "⚠️  Some tests failed. Check the errors above."
    exit 1
fi
