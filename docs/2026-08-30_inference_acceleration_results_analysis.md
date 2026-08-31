# 2026-08-30 推理加速实验结果分析

分析时间：2026-08-30 CST

本分析基于 `results/accel/`、`results/accel_followup/` 和 `results/accel_followup_clean/` 下的推理加速实验结果。由于初始实验启动过程中曾多次执行 launcher，顶层 `results/accel/launcher.log` 可能存在覆盖或混写；本报告以各实验目录内的 `run_metadata.json`、`eval.json` 和 `profile.json` 为准。手动补跑的 `E11_rerun1` 和 `E11_bf16_sdpa` 没有 `run_metadata.json`，只使用其 `eval.json` 和 `profile.json`。

## 输入与环境

- 模型：`/data2/home/models/Qwen3.8-27B`
- 测试集：`data/androidcontrol_mini/test.json`
- 完整评测：51 个 step
- 性能剖析：`--limit 5 --warmup 1`
- 点击容忍半径：`100.0`
- 初始成功实验的主要 GPU 设置：`CUDA_VISIBLE_DEVICES=1,6`
- E15-E18 干净补跑的 GPU 设置：`CUDA_VISIBLE_DEVICES=4,5`
- Python：`/data1/home/wuzheng/.conda/envs/qg/bin/python`
- launcher 记录的软件版本：
  - Python 3.11.15
  - torch 2.13.0
  - transformers 5.16.1
  - qwen-vl-utils 0.0.14
  - vLLM / SGLang 未安装

## 运行状态

| 实验 | 状态 | 备注 |
| --- | --- | --- |
| E00 | 中断 | 初始 baseline 尝试在 `py_compile` 后停止。 |
| E00_rerun1 | 中断 | 重复 launcher 运行，未产生完整结果。 |
| E00_rerun2 | 中断 | 重复 launcher 运行，未产生完整结果。 |
| E00_rerun3 | 成功 | 作为本轮复现加速 baseline。 |
| E01-E04 | 成功 | 完成解码长度上限和 SDPA 对比。 |
| E05 | 失败 | FlashAttention 2 包未安装。 |
| E06-E13 | 成功 | 完成 dtype、视觉 token 和静态 batch 实验。 |
| E14 | 跳过 | 未提供 `--serving_command`。 |
| E11_rerun1 | 成功 | 手动补跑 aggressive visual-token reduction。 |
| E11_bf16_sdpa | 成功 | 手动测试 aggressive visual-token reduction + BF16 + SDPA。 |
| E15-E18 | 成功 | 修复 batch decode 后完成动态视觉 token 与 batch 补跑。 |
| E15-E18 clean | 成功 | 在空闲 GPU 4,5 上补跑，E15-E18 的速度结论以这组为准。 |

中断的 E00 目录没有完整 `eval.json` / `profile.json`，不用于指标对比。

## 核心结论

当前最好的已完成单请求加速方案是 **E11**：

- 配置：`max_new_tokens=48`，`dtype=auto`，默认 attention，`batch_size=1`，`visual_token_mode=aggressive_reduce`
- `gui_only` 类型准确率：97.30%
- `gui_only` step 成功率：91.89%
- `strict` step 成功率：76.47%
- `pred_unknown`：0 / 51
- `hit_max_new_tokens`：0 / 51
- 完整评测平均延迟：2.646s；E00_rerun3 为 3.877s
- 完整评测吞吐：0.365 samples/sec；E00_rerun3 为 0.248 samples/sec
- profile 平均总耗时：2.067s；E00_rerun3 为 3.372s
- 峰值显存：GPU0 24.31 GB / GPU1 27.68 GB，低于 E00_rerun3 的 25.14 GB / 28.39 GB

解释：E11 在保持主口径 `gui_only` 准确率不变的同时，将完整评测平均延迟降低约 31.8%，将 profile 总耗时降低约 38.7%。在当前实验矩阵中，这是唯一明确成立的单请求加速收益。

E03、E10、E12 和 E13 不建议采用：

- E03 (`max_new_tokens=32`) 会截断输出：8 / 51 命中 token 上限，8 条预测变成 `UNKNOWN`，SCROLL 从 7 / 7 掉到 0 / 7。
- E10 (`mild_reduce`) 速度更快，但 CLICK step 成功率从 91.67% 降到 83.33%。
- E12/E13 是 batch decode 修复前的旧 batch 结果，虽然单样本速度变快，但输出健康度崩溃。E12 有 16 个 `UNKNOWN`；E13 有 27 个 `UNKNOWN`、2 个 think 标签和 4 次 token 上限命中。

非视觉 token 方向中，E08 的 profile 总耗时最好，但相对 E00_rerun3 只快约 1.4%。实际有效的加速主要来自视觉 token 降低，而不是 dtype 或 SDPA。

补跑结果进一步确认：

- `E11_rerun1` 复现了 E11 的准确率和输出健康度，profile 总耗时为 2.034s。
- `E11_bf16_sdpa` 也保持准确率，profile 总耗时为 2.025s；但相对 `E11_rerun1` 的差距小于 1%，不能把 BF16+SDPA 视为主要加速来源。
- `E15` 的 `dynamic_safe` 保持准确率，但因为 CLICK/LONG_PRESS 保留默认视觉 token，输入 token 更多，速度慢于固定 `aggressive_reduce`。
- `E16` 的 `dynamic_aggressive` 不可接受，CLICK step 成功率降到 83.33%。
- `E17/E18` 证明 batch decode 修复有效：两者均为 0 UNKNOWN、0 think 标签、0 token 上限命中。在空闲 GPU 4,5 上，E18 是可考虑的吞吐候选，但显存压力较高。

## 补跑结果更新

`results/accel_followup/` 中的补跑发生在两项代码改动之后：

- batch 预处理改为 left padding，并显式设置 generation 的 pad/eos token id。
- 视觉 token 选择新增 `dynamic_safe` 和 `dynamic_aggressive`。

最早一批补跑变慢主要是原 GPU 被占用导致的资源竞争。后续分析应使用修正后的 `E11_rerun1` / `E11_bf16_sdpa`，以及在 GPU 4,5 上干净补跑的 E15-E18。当前本地工作区未看到 `results/accel_followup_clean/` 下的 E19/E20，因此“batch + 固定 aggressive visual token”是否优于 E18 仍未完成验证。

| ID | 配置 | GUI 类型准确率 | GUI step 成功率 | strict step 成功率 | unknown / cap hit | 平均输入 token | 评测延迟 | 评测吞吐 | profile 总耗时 | generate 平均 | 峰值显存 GB | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| E11_rerun1 | tok=48, dtype=auto, attn=default, bs=1, vis=aggressive_reduce | 97.30% | 91.89% | 76.47% | 0 / 0 | 1231 | 2.600s | 0.372 | 2.034s | 1.944s | 24.31 / 27.68 | 准确率和速度可复现 |
| E11_bf16_sdpa | tok=48, dtype=bfloat16, attn=sdpa, bs=1, vis=aggressive_reduce | 97.30% | 91.89% | 76.47% | 0 / 0 | 1231 | 2.590s | 0.373 | 2.025s | 1.936s | 24.31 / 27.68 | 略快但接近噪声 |
| E15 clean | tok=48, dtype=auto, attn=default, bs=1, vis=dynamic_safe | 97.30% | 91.89% | 76.47% | 0 / 0 | 2549 | 3.361s | 0.287 | 2.579s | 2.464s | 25.13 / 28.38 | 准确但慢于 E11 |
| E16 clean | tok=48, dtype=auto, attn=default, bs=1, vis=dynamic_aggressive | 97.30% | 86.49% | 72.55% | 0 / 0 | 1348 | 2.714s | 0.356 | 2.084s | 1.989s | 24.38 / 27.74 | 拒绝 |
| E17 clean | tok=48, dtype=auto, attn=default, bs=2, vis=dynamic_safe | 97.30% | 91.89% | 76.47% | 0 / 0 | 2549 | 2.668s | 0.357 | 2.326s | 2.211s | 26.54 / 29.63 | decode 已修复，收益中性 |
| E18 clean | tok=48, dtype=auto, attn=default, bs=4, vis=dynamic_safe | 97.30% | 91.89% | 76.47% | 0 / 0 | 2549 | 2.270s | 0.418 | 2.396s | 2.280s | 29.28 / 32.08 | 可作为吞吐候选 |

补跑结论：

- GPU 竞争消除后，E11 的准确率和速度稳定：`gui_only` type、`gui_only` step、strict step、输入 token 和输出健康度均一致。
- 在 aggressive visual-token reduction 基础上叠加 BF16 + SDPA 没有带来实质速度收益。它在 eval latency 和 profile total 上略快，但差距低于 1%。
- `dynamic_safe` 能保准确率，但单样本场景慢于固定 `aggressive_reduce`。
- `dynamic_aggressive` 会重复 E10 的 CLICK 定位退化，不应采用。
- batch decode 修复已经通过 E17/E18 验证：两者 `pred_unknown=0`、`contains_think_end=0`，且没有 token 上限命中。
- E18 在完整评测吞吐上优于 E11_rerun1 和 E11_bf16_sdpa，但显存明显增加，且 5-step profile 不同样占优。因此 E18 是吞吐候选，不是单请求延迟最优解。

## 总体对比

速度提升基准使用 `E00_rerun3`。

| ID | 配置 | 状态 | GUI 类型准确率 | GUI step 成功率 | strict step 成功率 | unknown / cap hit | 平均输入 token | 评测延迟 | 评测吞吐 | profile 总耗时 | profile 加速比 | 峰值显存 GB | 判断 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| E00_rerun3 | tok=128, dtype=auto, attn=default, bs=1, vis=default | 成功 | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 3.877s | 0.248 | 3.372s | 1.00x | 25.14 / 28.39 | 基线 |
| E01 | tok=64, dtype=auto, attn=default, bs=1, vis=default | 成功 | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 3.898s | 0.247 | 3.339s | 1.01x | 25.14 / 28.39 | 安全上限检查 |
| E02 | tok=48, dtype=auto, attn=default, bs=1, vis=default | 成功 | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 3.959s | 0.243 | 3.352s | 1.01x | 25.14 / 28.39 | 默认 decode cap |
| E03 | tok=32, dtype=auto, attn=default, bs=1, vis=default | 成功 | 75.68% | 70.27% | 58.82% | 8 / 8 | 3404 | 3.738s | 0.257 | 3.352s | 1.01x | 25.14 / 28.39 | 拒绝 |
| E04 | tok=48, dtype=auto, attn=sdpa, bs=1, vis=default | 成功 | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 3.890s | 0.247 | 3.358s | 1.00x | 25.14 / 28.39 | 中性 |
| E05 | tok=48, dtype=auto, attn=flash_attention_2, bs=1, vis=default | 失败 | - | - | - | - | - | - | - | - | - | - | 缺依赖 |
| E06 | tok=48, dtype=bfloat16, attn=default, bs=1, vis=default | 成功 | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 3.886s | 0.248 | 3.329s | 1.01x | 25.14 / 28.39 | 中性 |
| E07 | tok=48, dtype=float16, attn=default, bs=1, vis=default | 成功 | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 4.060s | 0.237 | 3.536s | 0.95x | 25.14 / 28.39 | 拒绝 |
| E08 | tok=48, dtype=bfloat16, attn=sdpa, bs=1, vis=default | 成功 | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 3.922s | 0.246 | 3.325s | 1.01x | 25.14 / 28.39 | 非视觉方向最好 |
| E09 | tok=48, dtype=float16, attn=sdpa, bs=1, vis=default | 成功 | 97.30% | 91.89% | 74.51% | 0 / 0 | 3404 | 4.021s | 0.239 | 3.552s | 0.95x | 25.14 / 28.39 | 拒绝 |
| E10 | tok=48, dtype=auto, attn=default, bs=1, vis=mild_reduce | 成功 | 97.30% | 86.49% | 72.55% | 0 / 0 | 1430 | 2.840s | 0.340 | 2.214s | 1.52x | 24.38 / 27.74 | 拒绝 |
| E11 | tok=48, dtype=auto, attn=default, bs=1, vis=aggressive_reduce | 成功 | 97.30% | 91.89% | 76.47% | 0 / 0 | 1231 | 2.646s | 0.365 | 2.067s | 1.63x | 24.31 / 27.68 | 保留 |
| E12 | tok=48, dtype=auto, attn=default, bs=2, vis=default | 成功 | 67.57% | 62.16% | 52.94% | 16 / 0 | 3404 | 2.767s | 0.343 | 2.835s | 1.19x | 26.54 / 29.63 | 拒绝 |
| E13 | tok=48, dtype=auto, attn=default, bs=4, vis=default | 成功 | 40.54% | 37.84% | 35.29% | 27 / 4 | 3404 | 2.348s | 0.402 | 2.582s | 1.31x | 29.29 / 32.08 | 拒绝 |
| E14 | serving backend | 跳过 | - | - | - | - | - | - | - | - | - | - | 未运行 |

## 解码长度上限

E00-E03 回答了计划中的第一个问题：最低安全 `max_new_tokens` 是 **48**，不是 32。

| ID | max_new_tokens | GUI step 成功率 | pred_unknown | cap hits | 输出最大 token | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E00_rerun3 | 128 | 91.89% | 0 | 0 | 42 | 安全 |
| E01 | 64 | 91.89% | 0 | 0 | 42 | 安全 |
| E02 | 48 | 91.89% | 0 | 0 | 42 | 安全 |
| E03 | 32 | 70.27% | 8 | 8 | 32 | 不安全 |

后续实验应继续使用 E02 的 `max_new_tokens=48` 作为默认解码上限。E03 无效，因为截断会造成 parser 失败和 action type 退化。最清晰的症状是 SCROLL：E02 保持 7 / 7，E03 掉到 0 / 7。

## 注意力实现与 dtype

E04-E09 中，所有成功的非 FP16 变体都保持了与 E02 相同的准确率：

- E04 SDPA：profile 总耗时 3.358s
- E06 BF16：profile 总耗时 3.329s
- E08 BF16 + SDPA：profile 总耗时 3.325s

E02、E04、E06、E08 之间的差异都很小；没有重复运行时，应视为接近噪声。E08 是这组里最快的，但相对 E02 只快约 0.8%，相对 E00_rerun3 的 profile 总耗时只快约 1.4%，不足以算作有意义的单独加速收益。

当前环境下 FP16 更慢：

- E07 profile 总耗时：3.536s
- E09 profile 总耗时：3.552s
- 准确率保持稳定，但延迟退化。

E05 在评测前失败，原因是 FlashAttention 2 未安装：

```text
ImportError: FlashAttention2 has been toggled on, but it cannot be used ... the package for FlashAttention2 doesn't seem to be installed.
```

这是环境依赖结论，不是模型质量结论。

## 视觉 token

视觉 token 降低是当前唯一带来明确延迟收益的已完成方向。

| ID | visual_token_mode | 平均输入 token | GUI step 成功率 | CLICK step 成功率 | 评测延迟 | profile 总耗时 | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E02 | default | 3404 | 91.89% | 91.67% | 3.959s | 3.352s | 参考 |
| E10 | mild_reduce | 1430 | 86.49% | 83.33% | 2.840s | 2.214s | 拒绝 |
| E11 | aggressive_reduce | 1231 | 91.89% | 91.67% | 2.646s | 2.067s | 保留 |

E10 的现象有些反直觉：它的输入 token 多于 E11，但 CLICK 定位更差。单个 mini 集不能过度解释这个差异，但它足够说明 E10 不满足当前验收标准。

E11 在 mini 集上可用：

- 保持 `gui_only.step_success_rate`。
- CLICK step 成功率仍是 22 / 24，与 E02 和 E00_rerun3 相同。
- SCROLL、TYPE、PRESS_BACK 均保持 100%。
- `pred_unknown` 保持为 0。
- 没有输出命中 48-token 上限。
- 峰值显存相对 E00_rerun3 降低约 GPU0 0.83 GB、GPU1 0.71 GB。

strict 分数从 74.51% 提升到 76.47%，原因是 OPEN_APP 也从 50.00% 提升到 66.67%。但 OPEN_APP 仍是环境级动作，因此这个提升只能作为次要观察。

## 批处理

旧 E12/E13 是 batch decode 修复前的结果，不可作为可用方案。

| ID | batch size | GUI 类型准确率 | GUI step 成功率 | pred_unknown | think 标签 | cap hits | profile 总耗时 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E12 | 2 | 67.57% | 62.16% | 16 | 1 | 0 | 2.835s | 26.54 / 29.63 GB |
| E13 | 4 | 40.54% | 37.84% | 27 | 2 | 4 | 2.582s | 29.29 / 32.08 GB |

旧 E12/E13 虽然改善了单样本 timing，但输出健康度崩溃。多个 `UNKNOWN` 样本的 `raw_response` 为空，同时仍记录了非零输出 token；E13 还产生了以 `</think>` 开头并截断的不完整 JSON。这说明问题来自 batch generation/decoding 路径，而不是纯模型质量下降。

GPU 4,5 上的 E17/E18 干净补跑说明 batch decode 修复有效：

| ID | batch size | visual mode | GUI step 成功率 | pred_unknown | think 标签 | cap hits | 评测吞吐 | profile 总耗时 | 峰值显存 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E17 clean | 2 | dynamic_safe | 91.89% | 0 | 0 | 0 | 0.357 | 2.326s | 26.54 / 29.63 GB |
| E18 clean | 4 | dynamic_safe | 91.89% | 0 | 0 | 0 | 0.418 | 2.396s | 29.28 / 32.08 GB |

E17 和 E18 保持了完整 `gui_only` 结果，并消除了旧 batch 的输出失败。E18 的完整评测吞吐达到 0.418 samples/sec，高于 `E11_rerun1` 的 0.372 和 `E11_bf16_sdpa` 的 0.373，说明干净环境下确实有吞吐收益。

代价是显存和 profile 子集的不确定性。E18 峰值显存达到 29.28 / 32.08 GB，明显高于 E11 的 24.31 / 27.68 GB。它的 5-step profile 总耗时也慢于 E11_rerun1，但完整评测吞吐更好。因此 E18 应被视为完整评测吞吐候选，而不是每请求延迟一定更优。

当前未解决的问题是：batch 与固定 `aggressive_reduce` 组合后，能否同时利用更低视觉 token 和修复后的 batch decode，进一步超过 E18。E19/E20 正是为这个问题设计的，但当前本地工作区还没有对应结果目录。

## 按动作类型

以下是选定实验的各动作 step 成功率：

| ID | CLICK | LONG_PRESS | OPEN_APP | PRESS_BACK | SCROLL | TYPE | WAIT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E00_rerun3 | 91.67% | 0.00% | 50.00% | 100.00% | 100.00% | 100.00% | 12.50% |
| E02 | 91.67% | 0.00% | 50.00% | 100.00% | 100.00% | 100.00% | 12.50% |
| E03 | 87.50% | 0.00% | 50.00% | 100.00% | 0.00% | 100.00% | 12.50% |
| E10 | 83.33% | 0.00% | 66.67% | 100.00% | 100.00% | 100.00% | 12.50% |
| E11 | 91.67% | 0.00% | 66.67% | 100.00% | 100.00% | 100.00% | 12.50% |
| E12 | 66.67% | 0.00% | 16.67% | 100.00% | 57.14% | 33.33% | 37.50% |
| E13 | 41.67% | 0.00% | 16.67% | 50.00% | 28.57% | 33.33% | 37.50% |
| E15 | 91.67% | 0.00% | 66.67% | 100.00% | 100.00% | 100.00% | 12.50% |
| E16 | 83.33% | 0.00% | 66.67% | 100.00% | 100.00% | 100.00% | 12.50% |
| E17 | 91.67% | 0.00% | 66.67% | 100.00% | 100.00% | 100.00% | 12.50% |
| E18 | 91.67% | 0.00% | 66.67% | 100.00% | 100.00% | 100.00% | 12.50% |

稳定模式与校准后 baseline 一致：

- CLICK 是最敏感的 GUI 质量指标。
- SCROLL 在安全解码上限和修复后的 batch 路径中保持稳定。
- LONG_PRESS 仍未支持或约束不足，持续失败。
- WAIT 在静态单帧评测中噪声很大，应继续排除在主要 `gui_only` 解释之外。

## 启动器与实验设计备注

一个实现细节影响了实验排序：`E10-E13` 虽然矩阵里写的是 `best-so-far`，但没有继承 E08 的 BF16+SDPA 配置，而是以 `dtype=auto` 和默认 attention 运行。

原因是 launcher 的验收阈值使用 `REF_GUI_ONLY_TYPE = 0.9730`，而实际测量值是 `36 / 37 = 0.9729729729`，显示时四舍五入为 97.30%。严格浮点比较会让这些与 baseline 等价的结果错过 viability gate。因此 fallback 的 best-so-far 配置仍是 `max_new_tokens=48`、`dtype=auto`、默认 attention。

这不影响 E11 结论，因为 E11 在其实际配置下已经更快且可用。后续 `E11_bf16_sdpa` 覆盖了“visual token + BF16 + SDPA”的组合问题，结果是准确率保持、速度只略快。

## 最终结论

1. `max_new_tokens=48` 是当前安全解码上限；`max_new_tokens=32` 不安全。
2. E11 是当前最好的简单单请求加速方案：`visual_token_mode=aggressive_reduce`，`batch_size=1`，`dtype=auto`，默认 attention。
3. `E11_bf16_sdpa` 略快于 `E11_rerun1`，但差距低于 1%；只有在 BF16/SDPA 已经符合运行约束时才值得采用。
4. E10 会降低 CLICK 定位成功率，应拒绝。
5. 旧 E12/E13 因为发生在 batch decode 修复前，不应采用；干净 E17/E18 证明 batch 输出健康度已修复，E18 是吞吐候选，但显存压力更高。
6. E05 只能在安装与当前 torch/CUDA 栈兼容的 FlashAttention 2 后重跑。
7. E14 未评测，因为没有提供 serving backend 命令。

## 建议补跑

下一步应验证 batch 与低视觉 token 设置能否叠加收益。当前本地没有 E19/E20 结果，因此仍建议完成或同步这两组实验。

先跑 batch size 2：

```bash
OUT_DIR=results/accel_followup_clean/E19_batch2_aggressive
mkdir -p "${OUT_DIR}"
CUDA_VISIBLE_DEVICES=4,5 python scripts/eval_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/eval.json" \
  --max_new_tokens 48 \
  --batch_size 2 \
  --visual_token_mode aggressive_reduce
CUDA_VISIBLE_DEVICES=4,5 python scripts/profile_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/profile.json" \
  --limit 5 \
  --warmup 1 \
  --max_new_tokens 48 \
  --batch_size 2 \
  --visual_token_mode aggressive_reduce
```

如果 batch size 2 保持 `pred_unknown=0` 且 CLICK 不退化，再跑 batch size 4：

```bash
OUT_DIR=results/accel_followup_clean/E20_batch4_aggressive
mkdir -p "${OUT_DIR}"
CUDA_VISIBLE_DEVICES=4,5 python scripts/eval_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/eval.json" \
  --max_new_tokens 48 \
  --batch_size 4 \
  --visual_token_mode aggressive_reduce
CUDA_VISIBLE_DEVICES=4,5 python scripts/profile_androidcontrol.py \
  --model_path /data2/home/models/Qwen3.8-27B \
  --test_json data/androidcontrol_mini/test.json \
  --output "${OUT_DIR}/profile.json" \
  --limit 5 \
  --warmup 1 \
  --max_new_tokens 48 \
  --batch_size 4 \
  --visual_token_mode aggressive_reduce
```
