#!/bin/bash
# Training script for MSU_LSTM

# Default configuration
DATA_DIR="src/data/DJIA/feature34-Inter-P532"
USE_ALL_FEATURES=true  # Set to false to use only SINGLE_FEATURE_IDX
SINGLE_FEATURE_IDX=0   # Only used when USE_ALL_FEATURES=false
LOSS="MSE"             # Loss function: "MSE" (default) or "MAE" (better for extreme predictions)
SEED=42                # Random seed for reproducibility
WINDOW_LEN=13
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
if [ "$LOSS" = "MAE" ]; then
    echo "Loss function: MAE (better for extreme predictions)"
    LOSS_ARG="--use_mae"
elif [ "$LOSS" = "MSE" ]; then
    echo "Loss function: MSE (default)"
    LOSS_ARG=""
else
    echo "ERROR: Invalid LOSS value '$LOSS'. Must be 'MSE' or 'MAE'."
    exit 1
fi
echo "Random seed: $SEED"
echo "Window length: $WINDOW_LEN weeks"
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
