#!/bin/bash

# Test validation and test with fixed rho values script
# Usage: bash test_val_fixed_rho.sh [experiment_prefix]
# Example: bash test_val_fixed_rho.sh outputs/1010/151526

if [ $# -eq 0 ]; then
    echo "Usage: bash test_val_fixed_rho.sh [experiment_prefix]"
    echo "Example: bash test_val_fixed_rho.sh outputs/1010/151526"
    exit 1
fi

EXPERIMENT_PREFIX=$1

echo "Testing experiment: $EXPERIMENT_PREFIX"
echo "============================================"

# Check if experiment directory exists
if [ ! -d "$EXPERIMENT_PREFIX" ]; then
    echo "Error: Experiment directory $EXPERIMENT_PREFIX does not exist"
    exit 1
fi

# Function to run validation tests
run_validation_tests() {
    echo ""
    echo "🔍 Running Validation Tests..."
    echo "----------------------------"

    # Validation with rho=1.0
    echo "📊 Validation with manual_rho=1.0..."
    python validate.py --prefix "$EXPERIMENT_PREFIX" --manual_rho 1.0
    if [ -f "$EXPERIMENT_PREFIX/json_file/val_results.json" ]; then
        mv "$EXPERIMENT_PREFIX/json_file/val_results.json" "$EXPERIMENT_PREFIX/json_file/val_results_rho_1.json"
        echo "✅ Results saved to: val_results_rho_1.json"
    fi

    # Validation with rho=0.0
    echo "📊 Validation with manual_rho=0.0..."
    python validate.py --prefix "$EXPERIMENT_PREFIX" --manual_rho 0.0
    if [ -f "$EXPERIMENT_PREFIX/json_file/val_results.json" ]; then
        mv "$EXPERIMENT_PREFIX/json_file/val_results.json" "$EXPERIMENT_PREFIX/json_file/val_results_rho_0.json"
        echo "✅ Results saved to: val_results_rho_0.json"
    fi

    # Validation with rho=0.5
    echo "📊 Validation with manual_rho=0.5..."
    python validate.py --prefix "$EXPERIMENT_PREFIX" --manual_rho 0.5
    if [ -f "$EXPERIMENT_PREFIX/json_file/val_results.json" ]; then
        mv "$EXPERIMENT_PREFIX/json_file/val_results.json" "$EXPERIMENT_PREFIX/json_file/val_results_rho_0.5.json"
        echo "✅ Results saved to: val_results_rho_0.5.json"
    fi

    # Validation with original MSU predictions
    echo "📊 Validation with original MSU predictions..."
    python validate.py --prefix "$EXPERIMENT_PREFIX"
    if [ -f "$EXPERIMENT_PREFIX/json_file/val_results.json" ]; then
        mv "$EXPERIMENT_PREFIX/json_file/val_results.json" "$EXPERIMENT_PREFIX/json_file/val_results_msu_original.json"
        echo "✅ Results saved to: val_results_msu_original.json"
    fi
}

# Function to run test tests
run_test_tests() {
    echo ""
    echo "🧪 Running Test Tests..."
    echo "---------------------"

    # Test with rho=1.0
    echo "🎯 Testing with manual_rho=1.0..."
    python test.py --prefix "$EXPERIMENT_PREFIX" --manual_rho 1.0
    if [ -f "$EXPERIMENT_PREFIX/json_file/test_results.json" ]; then
        mv "$EXPERIMENT_PREFIX/json_file/test_results.json" "$EXPERIMENT_PREFIX/json_file/test_results_rho_1.json"
        echo "✅ Results saved to: test_results_rho_1.json"
    fi

    # Test with rho=0.0
    echo "🎯 Testing with manual_rho=0.0..."
    python test.py --prefix "$EXPERIMENT_PREFIX" --manual_rho 0.0
    if [ -f "$EXPERIMENT_PREFIX/json_file/test_results.json" ]; then
        mv "$EXPERIMENT_PREFIX/json_file/test_results.json" "$EXPERIMENT_PREFIX/json_file/test_results_rho_0.json"
        echo "✅ Results saved to: test_results_rho_0.json"
    fi

    # Test with rho=0.5
    echo "🎯 Testing with manual_rho=0.5..."
    python test.py --prefix "$EXPERIMENT_PREFIX" --manual_rho 0.5
    if [ -f "$EXPERIMENT_PREFIX/json_file/test_results.json" ]; then
        mv "$EXPERIMENT_PREFIX/json_file/test_results.json" "$EXPERIMENT_PREFIX/json_file/test_results_rho_0.5.json"
        echo "✅ Results saved to: test_results_rho_0.5.json"
    fi

    # Test with original MSU predictions
    echo "🎯 Testing with original MSU predictions..."
    python test.py --prefix "$EXPERIMENT_PREFIX"
    if [ -f "$EXPERIMENT_PREFIX/json_file/test_results.json" ]; then
        mv "$EXPERIMENT_PREFIX/json_file/test_results.json" "$EXPERIMENT_PREFIX/json_file/test_results_msu_original.json"
        echo "✅ Results saved to: test_results_msu_original.json"
    fi
}

# Run both validation and test
run_validation_tests
run_test_tests

echo ""
echo "============================================"
echo "🎉 All tests completed!"
echo ""
echo "📁 Results available in $EXPERIMENT_PREFIX/json_file/:"
echo ""
echo "📊 Validation Results:"
echo "  - val_results_rho_1.json (100% investment)"
echo "  - val_results_rho_0.json (0% investment)"
echo "  - val_results_rho_0.5.json (50% investment)"
echo "  - val_results_msu_original.json (MSU predictions)"
echo ""
echo "🧪 Test Results:"
echo "  - test_results_rho_1.json (100% investment)"
echo "  - test_results_rho_0.json (0% investment)"
echo "  - test_results_rho_0.5.json (50% investment)"
echo "  - test_results_msu_original.json (MSU predictions)"
echo ""
echo "💡 Compare the performance metrics to see which rho strategy works best!"