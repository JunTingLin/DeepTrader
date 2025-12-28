#!/bin/bash
# Training script for MSU_TE1D (Transformer Encoder)

# Default configuration
DATA_DIR="src/data/DJIA/feature34-Inter-P532"
USE_ALL_FEATURES=false  # Set to false to use only SINGLE_FEATURE_IDX
SINGLE_FEATURE_IDX=9   # Only used when USE_ALL_FEATURES=false
SEED=42                # Random seed for reproducibility

# Loss function configuration
LOSS_FUNCTION="MSE"    # Options: MSE, MAE, Sharpe, IC, Corr, Combined
# For Combined loss only:
ALPHA=0.5              # Weight for MSE component (0.0-1.0)
BETA=0.5               # Weight for trend component (0.0-1.0)
TREND_LOSS_TYPE="sharpe"  # Options: sharpe, ic, corr

# Model architecture
WINDOW_LEN=13
TRAIN_STEP=1           # Step size for training ground truth (default: 1)
VAL_STEP=1            # Step size for validation ground truth (default: 21)
TEST_STEP=1           # Step size for test ground truth (default: 21)
HIDDEN_DIM=128         # Transformer hidden dimension
DEPTH=2                # Number of transformer layers
HEADS=4                # Number of attention heads (must match pretrained model!)
MLP_DIM=32             # MLP dimension in transformer (must match pretrained model!)
DIM_HEAD=4             # Dimension per attention head (must match pretrained model!)
DROPOUT=0.1            # Dropout rate (must match pretrained model!)
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
echo "Loss function: $LOSS_FUNCTION"
LOSS_ARG="--loss_function $LOSS_FUNCTION"

if [ "$LOSS_FUNCTION" = "Combined" ]; then
    echo "  Combined loss weights: alpha=$ALPHA (MSE), beta=$BETA ($TREND_LOSS_TYPE)"
    LOSS_ARG="$LOSS_ARG --alpha $ALPHA --beta $BETA --trend_loss_type $TREND_LOSS_TYPE"
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
