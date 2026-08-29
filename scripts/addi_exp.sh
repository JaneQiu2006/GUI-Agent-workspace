OUT_DIR=results/accel_followup/E11_rerun1
mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES=1,6 python scripts/eval_androidcontrol.py \
--model_path /data2/home/models/Qwen3.8-27B \
--test_json data/androidcontrol_mini/test.json \
--output "${OUT_DIR}/eval.json" \
--max_new_tokens 48 \
--visual_token_mode aggressive_reduce

CUDA_VISIBLE_DEVICES=1,6 python scripts/profile_androidcontrol.py \
--model_path /data2/home/models/Qwen3.8-27B \
--test_json data/androidcontrol_mini/test.json \
--output "${OUT_DIR}/profile.json" \
--limit 5 \
--warmup 1 \
--max_new_tokens 48 \
--visual_token_mode aggressive_reduce

# 2. 补跑 E11 + BF16 + SDPA

OUT_DIR=results/accel_followup/E11_bf16_sdpa
mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES=1,6 python scripts/eval_androidcontrol.py \
--model_path /data2/home/models/Qwen3.8-27B \
--test_json data/androidcontrol_mini/test.json \
--output "${OUT_DIR}/eval.json" \
--max_new_tokens 48 \
--dtype bfloat16 \
--attn_implementation sdpa \
--visual_token_mode aggressive_reduce

CUDA_VISIBLE_DEVICES=1,6 python scripts/profile_androidcontrol.py \
--model_path /data2/home/models/Qwen3.8-27B \
--test_json data/androidcontrol_mini/test.json \
--output "${OUT_DIR}/profile.json" \
--limit 5 \
--warmup 1 \
--max_new_tokens 48 \
--dtype bfloat16 \
--attn_implementation sdpa \
--visual_token_mode aggressive_reduce

# 3. 新增动态视觉 token 实验
# 这已经包含在 E15-E16：

CUDA_VISIBLE_DEVICES=1,6 python scripts/run_accel_experiments.py \
--experiments E15-E16 \
--output_root results/accel_followup \
--resume

# 4. batch decode 修复后的复测
# 这已经包含在 E17-E18：

CUDA_VISIBLE_DEVICES=1,6 python scripts/run_accel_experiments.py \
--experiments E17-E18 \
--output_root results/accel_followup \
--resume