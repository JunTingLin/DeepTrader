#!/bin/bash
# Training script for MSU_LSTM

# Default configuration
DATA_DIR="src/data/DJIA/feature34-Inter-2"
USE_ALL_FEATURES=true  # Set to false to use only SINGLE_FEATURE_IDX
SINGLE_FEATURE_IDX=0   # Only used when USE_ALL_FEATURES=false
SEED=42                # Random seed for reproducibility

# Loss function configuration
LOSS_FUNCTION="MSE"    # Options: MSE, MAE, Sharpe, IC, Corr, Combined
# For Combined loss only:
ALPHA=0.7              # Weight for MSE component (0.0-1.0)
BETA=0.3               # Weight for trend component (0.0-1.0)
TREND_LOSS_TYPE="sharpe"  # Options: sharpe, ic, corr

# Data parameters
WINDOW_LEN=13          # Window length in weeks (13 or 26, must match ground truth JSON)
TRAIN_STEP=21           # Step size for training ground truth (default: 1)
VAL_STEP=21            # Step size for validation ground truth (default: 21)
TEST_STEP=21           # Step size for test ground truth (default: 21)
HIDDEN_DIM=128
DROPOUT=0.5
EPOCHS=200
BATCH_SIZE=32
LR=0.0001
WEIGHT_DECAY=0.0001
PATIENCE=30
NUM_WORKERS=4

echo "=============================================================================="
echo "Training MSU_LSTM"
echo "=============================================================================="
echo "Data directory: $DATA_DIR"
if [ "$USE_ALL_FEATURES" = true ]; then
    echo "Features: ALL (27 features)"
    FEATURE_ARG=""
else
    echo "Features: Single feature (index $SINGLE_FEATURE_IDX)"
    FEATURE_ARG="--feature_idx $SINGLE_FEATURE_IDX"
fi

# Handle loss function argument
echo "Loss function: $LOSS_FUNCTION"
LOSS_ARG="--loss_function $LOSS_FUNCTION"

if [ "$LOSS_FUNCTION" = "Combined" ]; then
    echo "  Combined loss weights: alpha=$ALPHA (MSE), beta=$BETA ($TREND_LOSS_TYPE)"
    LOSS_ARG="$LOSS_ARG --alpha $ALPHA --beta $BETA --trend_loss_type $TREND_LOSS_TYPE"
fi
echo "Random seed: $SEED"
echo "Window length: $WINDOW_LEN weeks"
echo "Ground truth steps: train=$TRAIN_STEP, val=$VAL_STEP, test=$TEST_STEP"
echo "Hidden dim: $HIDDEN_DIM"
echo "Dropout: $DROPOUT"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Weight decay: $WEIGHT_DECAY"
echo "Early stopping patience: $PATIENCE"
echo "=============================================================================="

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Run training
python ./src/MSU/train_msu_lstm.py \
    --data_dir $DATA_DIR \
    $FEATURE_ARG \
    $LOSS_ARG \
    --seed $SEED \
    --train_step $TRAIN_STEP \
    --val_step $VAL_STEP \
    --test_step $TEST_STEP \
    --window_len $WINDOW_LEN \
    --hidden_dim $HIDDEN_DIM \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --patience $PATIENCE \
    --num_workers $NUM_WORKERS

echo ""
echo "Training complete!"
echo "Check ./src/MSU/checkpoints/ for saved models and tensorboard logs"
echo ""
echo "To view tensorboard, run:"
echo "  tensorboard --logdir ./src/MSU/checkpoints/<run_name>/tensorboard"
