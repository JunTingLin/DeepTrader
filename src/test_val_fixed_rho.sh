#!/bin/bash

# Test validation and test with all rho strategies script
# Usage: bash src/test_val_fixed_rho.sh [experiment_prefix]
# Example: bash src/test_val_fixed_rho.sh src/outputs/1010/151526
#
# This script tests the following rho strategies:
# 1. rho=1.0 (100% investment)
# 2. rho=0.0 (0% investment)
# 3. rho=0.5 (50% investment)
# 4. MSU original (model predicted rho from ASU training)
# 5. MSU predicted rho (supervised learning)
# 6. Random rho (random rho per step, seed=42)
# 7. Ground truth rho (perfect rho, performance upper bound)

if [ $# -eq 0 ]; then
    echo "Usage: bash src/test_val_fixed_rho.sh [experiment_prefix]"
    echo "Example: bash src/test_val_fixed_rho.sh src/outputs/1010/151526"
    exit 1
fi

EXPERIMENT_PREFIX=$1

# Ground truth file paths
VAL_GT_FILE="src/data/DJIA/feature34-Inter-P532/MSU_val_ground_truth_step21.json"
TEST_GT_FILE="src/data/DJIA/feature34-Inter-P532/MSU_test_ground_truth_step21.json"

# Random rho file paths
RANDOM_RHO_VAL_FILE="src/data/DJIA/feature34-Inter-P532/random_rho_val_seed42.json"
RANDOM_RHO_TEST_FILE="src/data/DJIA/feature34-Inter-P532/random_rho_test_seed42.json"

# MSU predicted rho file paths
MSU_PREDICTED_RHO_VAL_FILE="src/MSU/checkpoints/MSU_LSTM_20251218_200329/msu_predicted_rho_val.json"
MSU_PREDICTED_RHO_TEST_FILE="src/MSU/checkpoints/MSU_LSTM_20251218_200329/msu_predicted_rho_test.json"

echo "Testing experiment: $EXPERIMENT_PREFIX"
echo "============================================"

# Check if experiment directory exists
if [ ! -d "$EXPERIMENT_PREFIX" ]; then
    echo "Error: Experiment directory $EXPERIMENT_PREFIX does not exist"
    exit 1
fi

# Function to run validation tests
run_validation_tests() {
    # echo ""
    # echo "🔍 Running Validation Tests..."
    # echo "----------------------------"

    # # Validation with rho=1.0
    # echo "📊 Validation with manual_rho=1.0..."
    # python src/validate.py --prefix "$EXPERIMENT_PREFIX" --manual_rho 1.0
    # if [ -f "$EXPERIMENT_PREFIX/json_file/val_results.json" ]; then
    #     mv "$EXPERIMENT_PREFIX/json_file/val_results.json" "$EXPERIMENT_PREFIX/json_file/val_results_rho_1.json"
    #     echo "✅ Results saved to: val_results_rho_1.json"
    # fi

    # # Validation with rho=0.0
    # echo "📊 Validation with manual_rho=0.0..."
    # python src/validate.py --prefix "$EXPERIMENT_PREFIX" --manual_rho 0.0
    # if [ -f "$EXPERIMENT_PREFIX/json_file/val_results.json" ]; then
    #     mv "$EXPERIMENT_PREFIX/json_file/val_results.json" "$EXPERIMENT_PREFIX/json_file/val_results_rho_0.json"
    #     echo "✅ Results saved to: val_results_rho_0.json"
    # fi

    # # Validation with rho=0.5
    # echo "📊 Validation with manual_rho=0.5..."
    # python src/validate.py --prefix "$EXPERIMENT_PREFIX" --manual_rho 0.5
    # if [ -f "$EXPERIMENT_PREFIX/json_file/val_results.json" ]; then
    #     mv "$EXPERIMENT_PREFIX/json_file/val_results.json" "$EXPERIMENT_PREFIX/json_file/val_results_rho_0.5.json"
    #     echo "✅ Results saved to: val_results_rho_0.5.json"
    # fi

    # # Validation with original MSU predictions
    # echo "📊 Validation with original MSU predictions..."
    # python src/validate.py --prefix "$EXPERIMENT_PREFIX"
    # if [ -f "$EXPERIMENT_PREFIX/json_file/val_results.json" ]; then
    #     mv "$EXPERIMENT_PREFIX/json_file/val_results.json" "$EXPERIMENT_PREFIX/json_file/val_results_msu_original.json"
    #     echo "✅ Results saved to: val_results_msu_original.json"
    # fi

    # Validation with MSU predicted rho
    echo "📊 Validation with MSU predicted rho (supervised learning)..."
    if [ -f "$MSU_PREDICTED_RHO_VAL_FILE" ]; then
        python src/validate.py --prefix "$EXPERIMENT_PREFIX" --ground_truth_rho "$MSU_PREDICTED_RHO_VAL_FILE"
        if [ -f "$EXPERIMENT_PREFIX/json_file/val_results.json" ]; then
            mv "$EXPERIMENT_PREFIX/json_file/val_results.json" "$EXPERIMENT_PREFIX/json_file/val_results_msu_lstm_predicted.json"
            echo "✅ Results saved to: val_results_msu_lstm_predicted.json"
        fi
    else
        echo "⚠️  Warning: MSU predicted rho file not found: $MSU_PREDICTED_RHO_VAL_FILE"
        echo "    Skipping MSU validation test"
    fi

    # # Validation with random rho
    # echo "📊 Validation with random rho..."
    # if [ -f "$RANDOM_RHO_VAL_FILE" ]; then
    #     python src/validate.py --prefix "$EXPERIMENT_PREFIX" --ground_truth_rho "$RANDOM_RHO_VAL_FILE"
    #     if [ -f "$EXPERIMENT_PREFIX/json_file/val_results.json" ]; then
    #         mv "$EXPERIMENT_PREFIX/json_file/val_results.json" "$EXPERIMENT_PREFIX/json_file/val_results_random_rho.json"
    #         echo "✅ Results saved to: val_results_random_rho.json"
    #     fi
    # else
    #     echo "⚠️  Warning: Random rho file not found: $RANDOM_RHO_VAL_FILE"
    #     echo "    Skipping random rho validation test"
    # fi

    # # Validation with ground truth rho
    # echo "📊 Validation with ground truth rho (performance upper bound)..."
    # if [ -f "$VAL_GT_FILE" ]; then
    #     python src/validate.py --prefix "$EXPERIMENT_PREFIX" --ground_truth_rho "$VAL_GT_FILE"
    #     if [ -f "$EXPERIMENT_PREFIX/json_file/val_results.json" ]; then
    #         mv "$EXPERIMENT_PREFIX/json_file/val_results.json" "$EXPERIMENT_PREFIX/json_file/val_results_ground_truth_rho.json"
    #         echo "✅ Results saved to: val_results_ground_truth_rho.json"
    #     fi
    # else
    #     echo "⚠️  Warning: Ground truth file not found: $VAL_GT_FILE"
    #     echo "    Skipping ground truth validation test"
    # fi
}

# Function to run test tests
run_test_tests() {
    echo ""
    echo "🧪 Running Test Tests..."
    echo "---------------------"

    # # Test with rho=1.0
    # echo "🎯 Testing with manual_rho=1.0..."
    # python src/test.py --prefix "$EXPERIMENT_PREFIX" --manual_rho 1.0
    # if [ -f "$EXPERIMENT_PREFIX/json_file/test_results.json" ]; then
    #     mv "$EXPERIMENT_PREFIX/json_file/test_results.json" "$EXPERIMENT_PREFIX/json_file/test_results_rho_1.json"
    #     echo "✅ Results saved to: test_results_rho_1.json"
    # fi

    # # Test with rho=0.0
    # echo "🎯 Testing with manual_rho=0.0..."
    # python src/test.py --prefix "$EXPERIMENT_PREFIX" --manual_rho 0.0
    # if [ -f "$EXPERIMENT_PREFIX/json_file/test_results.json" ]; then
    #     mv "$EXPERIMENT_PREFIX/json_file/test_results.json" "$EXPERIMENT_PREFIX/json_file/test_results_rho_0.json"
    #     echo "✅ Results saved to: test_results_rho_0.json"
    # fi

    # # Test with rho=0.5
    # echo "🎯 Testing with manual_rho=0.5..."
    # python src/test.py --prefix "$EXPERIMENT_PREFIX" --manual_rho 0.5
    # if [ -f "$EXPERIMENT_PREFIX/json_file/test_results.json" ]; then
    #     mv "$EXPERIMENT_PREFIX/json_file/test_results.json" "$EXPERIMENT_PREFIX/json_file/test_results_rho_0.5.json"
    #     echo "✅ Results saved to: test_results_rho_0.5.json"
    # fi

    # # Test with original MSU predictions
    # echo "🎯 Testing with original MSU predictions..."
    # python src/test.py --prefix "$EXPERIMENT_PREFIX"
    # if [ -f "$EXPERIMENT_PREFIX/json_file/test_results.json" ]; then
    #     mv "$EXPERIMENT_PREFIX/json_file/test_results.json" "$EXPERIMENT_PREFIX/json_file/test_results_msu_original.json"
    #     echo "✅ Results saved to: test_results_msu_original.json"
    # fi

    # Test with MSU predicted rho
    echo "🎯 Testing with MSU predicted rho (supervised learning)..."
    if [ -f "$MSU_PREDICTED_RHO_TEST_FILE" ]; then
        python src/test.py --prefix "$EXPERIMENT_PREFIX" --ground_truth_rho "$MSU_PREDICTED_RHO_TEST_FILE"
        if [ -f "$EXPERIMENT_PREFIX/json_file/test_results.json" ]; then
            mv "$EXPERIMENT_PREFIX/json_file/test_results.json" "$EXPERIMENT_PREFIX/json_file/test_results_msu_lstm_predicted.json"
            echo "✅ Results saved to: test_results_msu_lstm_predicted.json"
        fi
    else
        echo "⚠️  Warning: MSU predicted rho file not found: $MSU_PREDICTED_RHO_TEST_FILE"
        echo "    Skipping MSU test"
    fi

    # # Test with random rho
    # echo "🎯 Testing with random rho..."
    # if [ -f "$RANDOM_RHO_TEST_FILE" ]; then
    #     python src/test.py --prefix "$EXPERIMENT_PREFIX" --ground_truth_rho "$RANDOM_RHO_TEST_FILE"
    #     if [ -f "$EXPERIMENT_PREFIX/json_file/test_results.json" ]; then
    #         mv "$EXPERIMENT_PREFIX/json_file/test_results.json" "$EXPERIMENT_PREFIX/json_file/test_results_random_rho.json"
    #         echo "✅ Results saved to: test_results_random_rho.json"
    #     fi
    # else
    #     echo "⚠️  Warning: Random rho file not found: $RANDOM_RHO_TEST_FILE"
    #     echo "    Skipping random rho test"
    # fi

    # # Test with ground truth rho
    # echo "🎯 Testing with ground truth rho (performance upper bound)..."
    # if [ -f "$TEST_GT_FILE" ]; then
    #     python src/test.py --prefix "$EXPERIMENT_PREFIX" --ground_truth_rho "$TEST_GT_FILE"
    #     if [ -f "$EXPERIMENT_PREFIX/json_file/test_results.json" ]; then
    #         mv "$EXPERIMENT_PREFIX/json_file/test_results.json" "$EXPERIMENT_PREFIX/json_file/test_results_ground_truth_rho.json"
    #         echo "✅ Results saved to: test_results_ground_truth_rho.json"
    #     fi
    # else
    #     echo "⚠️  Warning: Ground truth file not found: $TEST_GT_FILE"
    #     echo "    Skipping ground truth test"
    # fi
}

# Run both validation and test
run_validation_tests
run_test_tests

echo ""
echo "============================================"
echo "🎉 All tests completed!"
echo ""
echo "📁 Results available in $EXPERIMENT_PREFIX/json_file/:"
echo "  - val_results_rho_1.json (rho=1.0)"
echo "  - val_results_rho_0.json (rho=0.0)"
echo "  - val_results_rho_0.5.json (rho=0.5)"
echo "  - val_results_msu_original.json (ASU predicted rho)"
echo "  - val_results_msu_lstm_predicted.json (MSU predicted rho)"
echo "  - val_results_random_rho.json (random rho, if generated)"
echo "  - val_results_ground_truth_rho.json (perfect rho upper bound)"
echo ""
echo "  - test_results_rho_1.json (rho=1.0)"
echo "  - test_results_rho_0.json (rho=0.0)"
echo "  - test_results_rho_0.5.json (rho=0.5)"
echo "  - test_results_msu_original.json (ASU predicted rho)"
echo "  - test_results_msu_lstm_predicted.json (MSU predicted rho)"
echo "  - test_results_random_rho.json (random rho, if generated)"
echo "  - test_results_ground_truth_rho.json (perfect rho upper bound)"
echo ""
