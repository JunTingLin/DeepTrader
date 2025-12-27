#!/bin/bash
# Training script for MSU_TE1D (Transformer Encoder)

# Default configuration
DATA_DIR="src/data/DJIA/feature34-Inter-2"
USE_ALL_FEATURES=false  # Set to false to use only SINGLE_FEATURE_IDX
SINGLE_FEATURE_IDX=9   # Only used when USE_ALL_FEATURES=false
LOSS="MSE"             # Loss function: "MSE" (default) or "MAE" (better for extreme predictions)
SEED=42                # Random seed for reproducibility

# Model architecture
WINDOW_LEN=13
TRAIN_STEP=1           # Step size for training ground truth (default: 1)
VAL_STEP=1            # Step size for validation ground truth (default: 21)
TEST_STEP=1           # Step size for test ground truth (default: 21)
HIDDEN_DIM=128         # Transformer hidden dimension
DEPTH=2                # Number of transformer layers
HEADS=8                # Number of attention heads (increased from 4)
MLP_DIM=256            # MLP dimension in transformer (increased from 32)
DIM_HEAD=16            # Dimension per attention head (increased from 4)
DROPOUT=0.2            # Dropout rate (increased from 0.1)
EMB_DROPOUT=0.1        # Embedding dropout rate
MLP_HIDDEN_DIM=128     # Hidden dimension for prediction MLP head

# Training parameters
EPOCHS=200
BATCH_SIZE=32
LR=0.0001
WEIGHT_DECAY=0.0001
PATIENCE=30
NUM_WORKERS=4

# Pre-trained encoder (optional)
# Set to path of Stage 1 PMSU checkpoint to use pre-trained weights
# Example: PRETRAINED_ENCODER="src/MSU/checkpoints/msu_stage1_masked/0101/120000/best_model.pth"
PRETRAINED_ENCODER=""

echo "=============================================================================="
echo "Training MSU_TE1D (Transformer Encoder)"
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

# Handle pre-trained encoder
if [ -n "$PRETRAINED_ENCODER" ]; then
    echo "Pre-trained encoder: $PRETRAINED_ENCODER"
    PRETRAINED_ARG="--pretrained_encoder $PRETRAINED_ENCODER"
else
    echo "Pre-trained encoder: None (training from scratch)"
    PRETRAINED_ARG=""
fi

echo "Random seed: $SEED"
echo "Ground truth steps: train=$TRAIN_STEP, val=$VAL_STEP, test=$TEST_STEP"
echo "=============================================================================="
echo "Model Architecture:"
echo "  Window length: $WINDOW_LEN weeks"
echo "  Hidden dim: $HIDDEN_DIM"
echo "  Depth (layers): $DEPTH"
echo "  Attention heads: $HEADS"
echo "  MLP dim: $MLP_DIM"
echo "  Dim per head: $DIM_HEAD"
echo "  Dropout: $DROPOUT"
echo "  Embedding dropout: $EMB_DROPOUT"
echo "  MLP hidden dim: $MLP_HIDDEN_DIM"
echo "=============================================================================="
echo "Training Parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Early stopping patience: $PATIENCE"
echo "=============================================================================="

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Run training
python ./src/MSU/train_msu_te1d.py \
    --data_dir $DATA_DIR \
    $FEATURE_ARG \
    $LOSS_ARG \
    $PRETRAINED_ARG \
    --seed $SEED \
    --train_step $TRAIN_STEP \
    --val_step $VAL_STEP \
    --test_step $TEST_STEP \
    --window_len $WINDOW_LEN \
    --hidden_dim $HIDDEN_DIM \
    --depth $DEPTH \
    --heads $HEADS \
    --mlp_dim $MLP_DIM \
    --dim_head $DIM_HEAD \
    --dropout $DROPOUT \
    --emb_dropout $EMB_DROPOUT \
    --mlp_hidden_dim $MLP_HIDDEN_DIM \
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
echo "  tensorboard --logdir ./src/MSU/checkpoints/<run_name>"
