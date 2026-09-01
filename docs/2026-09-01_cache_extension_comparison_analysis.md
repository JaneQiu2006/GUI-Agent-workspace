# 2026-09-01 Cache Extension 对比实验分析

分析时间：2026-09-01 CST

本分析基于本地已同步的：

- `results/cache_extension_comparison/`
- `results/cache_extension_comparison_debug/`

实验目标是以 E11 为 baseline，对比：

| ID | 配置 |
| --- | --- |
| `E11_baseline` | `max_new_tokens=48`、`visual_token_mode=aggressive_reduce`、不启用 cache |
| `E11_page_baseline` | E11 + `page_cache_mode=inputs` + `page_cache_similarity=exact` |
| `E11_patch_extension` | E11 + `page_cache_mode=inputs` + `page_cache_similarity=tile`，记录 patch diff / changed bbox / stable tile hashes / patch risk gating |

profile 计划使用 `--generation_profile_mode manual_greedy`，因此预期包含 `prefill_seconds` 和 `ttft_seconds`。

## 当前结果状态

这轮对比实验没有产生有效模型效果结果。三组都在 full eval 阶段失败，没有进入 `profile_ttft` 阶段，因此没有可比较的准确率、延迟、TTFT 或 cache hit 指标。

| 结果目录 | 实验 | 状态 | 已完成步骤 | 失败步骤 | 结论 |
| --- | --- | --- | --- | --- | --- |
| `results/cache_extension_comparison/E11_baseline_rerun1` | E11 baseline | failed | `py_compile` | `eval` | 无有效 eval/profile |
| `results/cache_extension_comparison/E11_page_baseline` | E11 + Page-level | failed | `py_compile` | `eval` | 无有效 eval/profile |
| `results/cache_extension_comparison/E11_patch_extension` | E11 + Patch-level | failed | `py_compile` | `eval` | 无有效 eval/profile |
| `results/cache_extension_comparison_debug/E11_baseline` | E11 baseline | failed | `py_compile` | `eval` | 明确 CUDA OOM |
| `results/cache_extension_comparison_debug/E11_page_baseline` | E11 + Page-level | failed | `py_compile` | `eval` | 明确 CUDA OOM |
| `results/cache_extension_comparison_debug/E11_patch_extension` | E11 + Patch-level | failed | `py_compile` | `eval` | 明确 CUDA OOM |

`results/cache_extension_comparison/E11_baseline` 是一次早期残留，`run_metadata.json` 仍为 `running` 且只完成 `py_compile`，不纳入分析。

## 失败原因

`results/cache_extension_comparison_debug/*/stderr.log` 显示三组失败原因一致：

```text
torch.AcceleratorError: CUDA error: out of memory
```

OOM 发生在模型 forward 中的 Qwen3.5 language model / linear attention 路径：

```text
transformers/models/qwen3_5/modeling_qwen3_5.py
torch_chunk_gated_delta_rule
l2norm
torch.rsqrt((x * x).sum(...))
```

典型日志：

```text
memory allocation failed with OOM on device 1 while trying to allocate 44040192 bytes
free: 39059456, total: 85093777408
```

运行环境中 `CUDA_VISIBLE_DEVICES=0,6`，日志里的 `device 1` 对应可见设备列表中的第二张卡，也就是物理 GPU 6。该卡在失败时只剩约 39 MB free，连约 44 MB 的临时分配都无法满足。

## 对实验结论的影响

这次结果不能用于判断 Page-level baseline 或 Patch-level extension 的模型效果，因为：

- `E11_baseline` 在 `page_cache_mode=off` 时同样 OOM。
- 三组都没有完成 full eval，所以没有 `eval.json`。
- 三组都没有进入 `profile_ttft`，所以没有 `profile.json`，也没有 `prefill_seconds` / `ttft_seconds`。
- OOM 发生在公共模型 forward 路径，不是 cache lookup、processor input cache 或 patch diff 逻辑中的异常。

因此当前有效结论应更新为：

1. E11 cache 对比实验启动链路和参数生成正常，`py_compile` 通过。
2. 本次 `GPU 0,6` 资源条件不足，导致 E11 baseline 自身无法完成 eval。
3. 目前尚未获得 Page-level baseline 或 Patch-level extension 相对 E11 的准确率、延迟、TTFT 对比。
4. 不能把本次失败解释为 cache extension 引入的质量或性能回退。

## 与既有 E11 结果的关系

旧分析中 E11 的有效结果仍然成立，因为此前 E11 / E11_rerun1 在较空闲 GPU 上完成过：

- `gui_only` type accuracy：97.30%
- `gui_only` step success rate：91.89%
- `pred_unknown`：0 / 51
- `hit_max_new_tokens`：0 / 51
- profile total 约 2.03-2.07s

本次失败只是说明 `CUDA_VISIBLE_DEVICES=0,6` 在运行时不可用或显存碎片/占用过高，不推翻已有 E11 baseline。

## 建议重跑

优先使用此前验证过较稳定的 GPU 组合，例如 `4,5`，或先用 `nvidia-smi` 确认两张卡均有足够空闲显存。

```bash
python scripts/run_cache_extension_comparison.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output_root results/cache_extension_comparison_gpu45 \
  --gpus 4,5 \
  --profile_limit 5 \
  --warmup 1
```

如果仍然 OOM，应先单独重跑 E11 baseline 的 eval，确认模型本身能在当前 GPU 组合上运行：

```bash
CUDA_VISIBLE_DEVICES=4,5 python scripts/eval_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output results/cache_extension_comparison_gpu45/e11_eval_smoke.json \
  --max_new_tokens 48 \
  --visual_token_mode aggressive_reduce
```

只有 E11 baseline full eval 能完成后，Page-level 和 Patch-level 的对比才有解释价值。

## 后续判读口径

重跑成功后，重点比较：

- 准确率：`gui_only.type_accuracy`、`gui_only.step_success_rate`、CLICK step success。
- 输出健康度：`pred_unknown`、`hit_max_new_tokens`、`contains_think_end`。
- TTFT/profile：`prefill_seconds`、`ttft_seconds`、`generate_seconds`、`total_seconds`。
- cache：`processor_cache_hit_rate`、`page_cache_hit_types`、`patch_candidate_rate`、`patch_candidate_allowed_rate`。

预期第一版 Patch-level extension 只做 observe/gating，不复用局部 KV。因此如果 full eval 成功，模型输出和准确率应与 Page-level exact inputs 路径基本一致；它的价值主要体现在 patch candidate 覆盖率和风险分析，而不是直接降低 TTFT。
