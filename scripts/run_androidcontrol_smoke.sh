#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data2/home/models/Qwen3.8-27B}"
ANDROIDCONTROL_JSON="${ANDROIDCONTROL_JSON:?set ANDROIDCONTROL_JSON to AndroidControl json/jsonl annotation path}"
ANDROIDCONTROL_DATA_DIR="${ANDROIDCONTROL_DATA_DIR:?set ANDROIDCONTROL_DATA_DIR to the directory used to resolve image paths}"
LIMIT="${LIMIT:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/androidcontrol_smoke}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DEVICE="${DEVICE:-auto}"

mkdir -p "${OUTPUT_DIR}"

python scripts/prepare_androidcontrol_jsonl.py \
  --input "${ANDROIDCONTROL_JSON}" \
  --output "${OUTPUT_DIR}/androidcontrol_${LIMIT}.jsonl" \
  --limit "${LIMIT}"

python test_gui_benchmark.py \
  --model_path "${MODEL_PATH}" \
  --samples "${OUTPUT_DIR}/androidcontrol_${LIMIT}.jsonl" \
  --data_dir "${ANDROIDCONTROL_DATA_DIR}" \
  --output "${OUTPUT_DIR}/qwen38_baseline_results.jsonl" \
  --limit "${LIMIT}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --device "${DEVICE}"
