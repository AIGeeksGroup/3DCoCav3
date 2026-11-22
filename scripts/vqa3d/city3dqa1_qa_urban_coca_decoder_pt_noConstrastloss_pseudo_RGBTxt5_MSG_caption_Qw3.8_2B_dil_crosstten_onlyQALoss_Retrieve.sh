#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

set -e
set -o pipefail

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate Q3VL
fi
QWEN_PYTHON="${QWEN_PYTHON:-$(command -v python)}"

CONFIG="${CONFIG:-configs/finetune/city3dqa1_qa_urban_coca_decoder_pt.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/znn/UniScene3D_decoder}"
NOTE="${NOTE:-urban_decoder_pseudo_MSG_caption_Qw38_2B_retrie_caption_crossatt_lr10-6}"
EXP_NAME="${EXP_NAME:-city3dqa1_decoder_pseudo_MSG_caption_Qw38_2B_retrie_caption_crossattlr10-6}"

##/data1/znn/CODE_pk/UniScene3D/results/UniScene3D_City3DQA1_RGBtxt5_b128_Pretrain_City3DQA1_City3DQA1_UrbanTrainScenes_RGBtxt5_run1_0.0001/2026-08-17-17:12:12.293502/ckpt/ckpt_3000.pth

QA_MODE_DIR="${QA_MODE_DIR:-/data1/znn/Dataset_pk/CIty-3DQA/qa/urban_mode}"
QA_DIR="$(cd "${QA_MODE_DIR}/.." && pwd)"
QA_MODE_NAME="$(basename "${QA_MODE_DIR}")"
CACHE_BASE_DIR="${CACHE_BASE_DIR:-/data1/znn/Dataset_pk/CIty-3DQA/rendered_views_znn_piliang_pt}"
FGCLIP_FEATURE_DIR="${FGCLIP_FEATURE_DIR:-/data1/znn/Dataset_pk/CIty-3DQA/FG_CLIP_Feature/1scene_sameview_sce_caption_ckpt_6000}"
RENDER_BASE_DIR="${RENDER_BASE_DIR:-/data1/znn/Dataset_pk/CIty-3DQA/rendered_views_znn_piliang}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
BATCHSIZE="${BATCHSIZE:-96}"
BATCHSIZE_EVAL="${BATCHSIZE_EVAL:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_VIEWS="${NUM_VIEWS:-50}"
EPOCHS="${EPOCHS:-30}"
EPOCHS_PER_SAVE="${EPOCHS_PER_SAVE:-1}"
LR="${LR:-1e-6}"
RESUME="${RESUME:-False}"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
GPU_PER_NODE=4
NUM_NODES=1
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12345}"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${PROJECT_ROOT}/cache/huggingface_modules}"
mkdir -p "${HF_MODULES_CACHE}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${PROJECT_ROOT}/pretrained/Qwen3.8-2B-Distill}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/${EXP_NAME}_${NOTE}_${TIMESTAMP}.log"

echo "[INFO] Starting City3DQA1 QA-only training with retrieved captions, cached FG-CLIP features, and visual cross-attention"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Experiment output root: ${OUTPUT_ROOT}"
echo "[INFO] FG-CLIP feature directory: ${FGCLIP_FEATURE_DIR}"
echo "[INFO] Frozen Qwen base + trainable LoRA, scene adapter, and visual cross-attention: ${QWEN_MODEL_PATH}"
echo "[INFO] QA split directory: ${QA_MODE_DIR}"
echo "[INFO] PT cache directory: ${CACHE_BASE_DIR}"
echo "[INFO] sce_txt scene caption directory: ${RENDER_BASE_DIR}"
echo "[INFO] QA supervision only; one question-relevant scene caption is retrieved for each QA in train, val and test"
echo "[INFO] Batch size per process: ${BATCHSIZE}"
echo "[INFO] Eval batch size per process: ${BATCHSIZE_EVAL}"
echo "[INFO] Views per scene-question sample: ${NUM_VIEWS}"
echo "[INFO] Epochs: ${EPOCHS}"
echo "[INFO] Epochs per periodic checkpoint save: ${EPOCHS_PER_SAVE}"
echo "[INFO] Decoder learning rate: ${LR}"
echo "[INFO] Resume: ${RESUME}"
echo "[INFO] DataLoader workers per process: ${NUM_WORKERS}"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO] GPUs: ${GPU_PER_NODE}"
echo "[INFO] Main process port: ${MAIN_PROCESS_PORT}"
echo "[INFO] WANDB_MODE: ${WANDB_MODE}"
echo "[INFO] Log: ${LOGFILE}"

for split in train val test; do
    if [[ ! -f "${QA_MODE_DIR}/${split}.json" ]]; then
        echo "[ERROR] Missing QA split file: ${QA_MODE_DIR}/${split}.json" >&2
        exit 1
    fi
done

if [[ ! -d "${CACHE_BASE_DIR}" ]]; then
    echo "[ERROR] Missing PT cache directory: ${CACHE_BASE_DIR}" >&2
    exit 1
fi

if [[ ! -d "${RENDER_BASE_DIR}" ]]; then
    echo "[ERROR] Missing rendered caption directory: ${RENDER_BASE_DIR}" >&2
    exit 1
fi

if [[ ! -d "${FGCLIP_FEATURE_DIR}" ]]; then
    echo "[ERROR] Missing FG-CLIP feature directory: ${FGCLIP_FEATURE_DIR}" >&2
    exit 1
fi

if ! find "${FGCLIP_FEATURE_DIR}" -mindepth 2 -maxdepth 2 -type f -name '*.pt' -print -quit | grep -q .; then
    echo "[ERROR] No City/Area FG-CLIP feature files found under ${FGCLIP_FEATURE_DIR}." >&2
    exit 1
fi

if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
    echo "[ERROR] Missing local Qwen model directory: ${QWEN_MODEL_PATH}" >&2
    echo "[ERROR] Download empero-ai/Qwen3.8-2B-Distill to this path before running (offline mode is enabled)." >&2
    exit 1
fi

if ! find "${QWEN_MODEL_PATH}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .; then
    echo "[ERROR] Qwen weight files are missing from ${QWEN_MODEL_PATH}." >&2
    exit 1
fi

TRANSFORMERS_VERSION="$("${QWEN_PYTHON}" -c 'import transformers; print(transformers.__version__)' 2>/dev/null || true)"
if [[ -z "${TRANSFORMERS_VERSION}" ]] || [[ "$("${QWEN_PYTHON}" -c 'from packaging.version import Version; import sys; print(int(Version(sys.argv[1]) >= Version("5.15.0")))' "${TRANSFORMERS_VERSION}" 2>/dev/null || true)" != "1" ]]; then
    echo "[ERROR] Qwen3.5 requires transformers>=5.15.0 and PEFT in the selected Python environment." >&2
    exit 1
fi

if ! "${QWEN_PYTHON}" -c 'import peft' >/dev/null 2>&1; then
    echo "[ERROR] PEFT is not installed in ${QWEN_PYTHON}; install it in the Q3VL environment before training." >&2
    exit 1
fi

"${QWEN_PYTHON}" launch.py --mode accelerate \
    --gpu_per_node "${GPU_PER_NODE}" \
    --num_nodes "${NUM_NODES}" \
    --port "${MAIN_PROCESS_PORT}" \
    --run_file run_city3dqa1_qa_msg_caption_qwen_visual_cross_attention.py \
    --config "${CONFIG}" \
    base_dir="${OUTPUT_ROOT}" \
    resume="${RESUME}" \
    pretrain_ckpt_path="" \
    data.City3DQA1QAUrbanCoCaDecoderPT.qa_dir="${QA_DIR}" \
    data.City3DQA1QAUrbanCoCaDecoderPT.mode_name="${QA_MODE_NAME}" \
    data.City3DQA1QAUrbanCoCaDecoderPT.cache_base_dir="${CACHE_BASE_DIR}" \
    data.train="[City3DQA1MSGQAUrbanPT]" \
    data.val="[City3DQA1MSGQAUrbanPT]" \
    data.test="[City3DQA1MSGQAUrbanPT]" \
    data_wrapper.train="City3DQA1QwenQARetrievedCaptionWrapper" \
    data_wrapper.val="City3DQA1QwenQARetrievedCaptionWrapper" \
    data_wrapper.test="City3DQA1QwenQARetrievedCaptionWrapper" \
    ++data.City3DQA1CaptionUrbanCoCaDecoderPT.qa_dir="${QA_DIR}" \
    ++data.City3DQA1CaptionUrbanCoCaDecoderPT.mode_name="${QA_MODE_NAME}" \
    ++data.City3DQA1CaptionUrbanCoCaDecoderPT.cache_base_dir="${CACHE_BASE_DIR}" \
    ++data.City3DQA1CaptionUrbanCoCaDecoderPT.render_base_dir="${RENDER_BASE_DIR}" \
    ++data.City3DQA1CaptionUrbanCoCaDecoderPT.pseudo_captions_per_scene=5 \
    ++data.City3DQA1CaptionUrbanCoCaDecoderPT.pseudo_views_per_caption=4 \
    ++data.City3DQA1CaptionUrbanCoCaDecoderPT.msg_pseudo_ratio=3 \
    ++data.City3DQA1CaptionUrbanCoCaDecoderPT.max_target_length=512 \
    ++data.City3DQA1CaptionUrbanCoCaDecoderPT.num_views="${NUM_VIEWS}" \
    ++data.City3DQA1CaptionUrbanCoCaDecoderPT.qa_condition_max_length=128 \
    model.name="City3DQA1QwenVisualCrossAttentionGeneration" \
    ++model.qwen_model_path="${QWEN_MODEL_PATH}" \
    ++model.feature_root="${FGCLIP_FEATURE_DIR}" \
    model.train_pm_encoder=False \
    model.loss_list="[City3DQA1CoCaDecoderGenerationLoss]" \
    model.vis_loss_list="[City3DQA1CoCaDecoderGenerationLoss]" \
    model.generation.max_new_tokens=256 \
    model.heads.head_list="[qwen_visual_cross_attention_head]" \
    ++model.heads.qwen_visual_cross_attention_head.name="City3DQA1QwenVisualCrossAttentionHead" \
    ++model.heads.qwen_visual_cross_attention_head.lr="${LR}" \
    ++model.heads.qwen_visual_cross_attention_head.args.model_path="${QWEN_MODEL_PATH}" \
    ++model.heads.qwen_visual_cross_attention_head.args.scene_dim=512 \
    ++model.heads.qwen_visual_cross_attention_head.args.lora_rank=16 \
    ++model.heads.qwen_visual_cross_attention_head.args.lora_alpha=32 \
    ++model.heads.qwen_visual_cross_attention_head.args.lora_dropout=0.05 \
    ++model.heads.qwen_visual_cross_attention_head.args.cross_attention_heads=8 \
    ++model.heads.qwen_visual_cross_attention_head.args.cross_attention_dropout=0.0 \
    ++model.heads.qwen_visual_cross_attention_head.args.cross_attention_gate_init=0.1 \
    trainer="City3DQA1CoCaDecoderFrozenPMEvalTrainer" \
    ++eval.split="test" \
    solver.epochs_per_eval=1 \
    data.City3DQA1QAUrbanCoCaDecoderPT.max_target_length=64 \
    data.City3DQA1QAUrbanCoCaDecoderPT.max_question_length=128 \
    data.City3DQA1QAUrbanCoCaDecoderPT.num_views="${NUM_VIEWS}" \
    num_views="${NUM_VIEWS}" \
    data.City3DQA1QAUrbanCoCaDecoderPT.num_views="${NUM_VIEWS}" \
    logger.entity="${WANDB_ENTITY}" \
    ++logger.init_timeout="${WANDB_INIT_TIMEOUT}" \
    dataloader.batchsize="${BATCHSIZE}" \
    dataloader.batchsize_eval="${BATCHSIZE_EVAL}" \
    solver.lr="${LR}" \
    model.heads.qwen_visual_cross_attention_head.lr="${LR}" \
    solver.epochs="${EPOCHS}" \
    solver.epochs_per_save="${EPOCHS_PER_SAVE}" \
    dataloader.num_workers="${NUM_WORKERS}" \
    note="${NOTE}" \
    name="${EXP_NAME}" 2>&1 | tee "${LOGFILE}"
