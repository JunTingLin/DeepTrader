#!/bin/bash
# Training script for ASU_TE2D Supervised Pre-training

# Same configuration as GCN version, just use different model

DATA_DIR="src/data/DJIA/feature34-Inter-P532"
SEED=42
LOSS_FUNCTION="Combined"
ALPHA=0.8
BETA=0.2
RANKING_LOSS_TYPE="pairwise"
REGRESSION_LOSS_TYPE="mse"
MARGIN=0.1
MIN_ROR_DIFF=0.01
WINDOW_LEN=13
HORIZON=21
TRAIN_STRIDE=1
VAL_STRIDE=21  # Match trade_len for non-overlapping evaluation
TEST_STRIDE=21  # Match trade_len for non-overlapping evaluation
TRAIN_IDX=0
TRAIN_IDX_END=1304
VAL_IDX=1304
TEST_IDX=2087
TEST_IDX_END=2673
HIDDEN_DIM=128
NUM_BLOCKS=4
KERNEL_SIZE=2
DROPOUT=0.5
SPATIAL_BOOL=false
ADDAPTIVEADJ=true
EPOCHS=100
BATCH_SIZE=32
LR=0.0001
OPTIMIZER="adam"
WEIGHT_DECAY=0.0001
PATIENCE=20
SAVE_INTERVAL=10
NUM_WORKERS=4

echo "=============================================================================="
echo "Training ASU_TE2D with Supervised Learning"
echo "=============================================================================="
echo "Model: TCN + TE_2D (Transformer) + Spatial Attention"
echo "Loss: $LOSS_FUNCTION (alpha=$ALPHA, beta=$BETA)"
echo "=============================================================================="

SPATIAL_ARG=""
[ "$SPATIAL_BOOL" = true ] && SPATIAL_ARG="--spatial_bool"

ADDAPTIVEADJ_ARG=""
[ "$ADDAPTIVEADJ" = true ] && ADDAPTIVEADJ_ARG="--addaptiveadj"

python ./src/ASU/train_asu_te2d.py \
    --data_dir $DATA_DIR \
    --train_idx $TRAIN_IDX --train_idx_end $TRAIN_IDX_END \
    --val_idx $VAL_IDX --test_idx $TEST_IDX --test_idx_end $TEST_IDX_END \
    --window_len $WINDOW_LEN --horizon $HORIZON \
    --train_stride $TRAIN_STRIDE --val_stride $VAL_STRIDE --test_stride $TEST_STRIDE \
    --hidden_dim $HIDDEN_DIM --num_blocks $NUM_BLOCKS --kernel_size $KERNEL_SIZE \
    --dropout $DROPOUT $SPATIAL_ARG $ADDAPTIVEADJ_ARG \
    --loss_function $LOSS_FUNCTION --margin $MARGIN --min_ror_diff $MIN_ROR_DIFF \
    --alpha $ALPHA --beta $BETA \
    --ranking_loss_type $RANKING_LOSS_TYPE --regression_loss_type $REGRESSION_LOSS_TYPE \
    --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY --patience $PATIENCE --save_interval $SAVE_INTERVAL \
    --seed $SEED --num_workers $NUM_WORKERS --use_gpu

echo ""
echo "Training complete! Check ./src/ASU/checkpoints/ for results."
