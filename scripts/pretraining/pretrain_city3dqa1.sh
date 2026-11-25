#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="configs/city3dqa1_pretrain_pt.yaml"
NOTE="City3DQA1_UrbanTrainScenes_RGBtxt5_run1"
EXP_NAME="UniScene3D_City3DQA1_RGBtxt5"
LR="1e-4"

cd "${PROJECT_ROOT}"
set -e
set -o pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${EXP_NAME}_${NOTE}_${TIMESTAMP}.log"
EXP_DIR="results/${EXP_NAME}_${NOTE}"

echo "[INFO] Starting City3DQA1 pretraining: $EXP_NAME ($NOTE)"
echo "[INFO] Using QA split scene filter: /data1/znn/Dataset_pk/CIty-3DQA/qa/urban_mode/train.json"
echo "[INFO] Expected train cities: Longhua, Qingdao, Wuhu, Yingrenshi"
echo "[INFO] Dataset root: /data1/znn/Dataset_pk/CIty-3DQA/rendered_views_znn_piliang"
echo "[INFO] Config: $CONFIG"
echo "[INFO] Learning rate: $LR"
echo "[INFO] Logging to: $LOGFILE"
echo "[INFO] Experiment directory prefix: $EXP_DIR"

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GPU_PER_NODE="${GPU_PER_NODE:-4}"
export PORT="${PORT:-29511}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-$EXP_NAME}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$NOTE}"

echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] GPU_PER_NODE: $GPU_PER_NODE"
echo "[INFO] PORT: $PORT"
echo "[INFO] WANDB_MODE: $WANDB_MODE"
echo "[INFO] WANDB_PROJECT: $WANDB_PROJECT"

python launch.py \
    --mode accelerate \
    --run_file run_city3dqa1.py \
    --gpu_per_node "$GPU_PER_NODE" \
    --num_nodes 1 \
    --port "$PORT" \
    --config "$CONFIG" \
    logger.name=wandb \
    solver.lr="$LR" \
    naming_keywords="[dataloader.batchsize,task,note,solver.lr,time]" \
    note="$NOTE" \
    name="$EXP_NAME"
