# Visual Feature Locality Analysis

日期：2026-09-06

## 目标

本实验用于回答两层问题：

> 当 GUI 页面只有少量 patch 发生变化时，vision encoder 输出中是否也只有少量对应 feature 发生显著变化？

> merger / projector 后、真正进入 LLM prefix 前的 model-ready visual embeddings 是否仍保持空间局部性？

实验只做 profiling / analysis，不实现真正的 patch feature cache，也不训练或修改模型权重。后续 feature cache 可基于这里预留的边界继续实现：

```text
cached feature + changed patch recomputation + feature replacement
```

## 实验配置

主要结果目录：

```text
results/feature_locality_analysis/androidcontrol_1000_broad_100
results/feature_locality_analysis/androidcontrol_1000_broad_100_model_ready_visual
```

运行配置：

- 模型：`/data2/home/models/Qwen3.8-27B`
- 数据：`data/androidcontrol_1000/test.json`
- visual token mode：`aggressive_reduce`
- patch-level feature source：`model.visual`
- model-ready feature source：`model.visual.merger`
- feature layer：`final`
- feature metric：`cosine_distance`
- threshold：`0.005`, `0.01`, `0.03`, `0.05`, `0.1`
- page similarity：tile-based
- pair scope：dataset
- max pairs：100
- tile grid：`8 x 16`

两轮结果中：

- valid pair-layer rows：`100`
- error count：`0`
- hit type：`patch_candidate=55`, `near=45`
- patch-level visual feature tokens：`1508` 为主，少数 `1568`
- model-ready visual tokens：`377` 为主，少数 `392`
- model-ready visual token grid：`[29,13]` 为主，少数 `[28,14]`
- processor merge size：`2`
- model-ready alignment method：`tile_mask_center_sample_to_merged_visual_token_grid`

## Patch-Level 核心结果

Pixel-space patch 变化：

| metric | value |
|---|---:|
| mean `R_pixel` | `0.0611` |
| median `R_pixel` | `0.0547` |
| p90 `R_pixel` | `0.1180` |
| min / max | `0.0078 / 0.1484` |

Feature-space 显著变化比例：

| tau | mean `R_feature` | median | p90 | max |
|---:|---:|---:|---:|---:|
| `0.005` | `0.0315` | `0.0239` | `0.0777` | `0.1253` |
| `0.01` | `0.0205` | `0.0146` | `0.0513` | `0.0816` |
| `0.03` | `0.0075` | `0.0046` | `0.0200` | `0.0418` |
| `0.05` | `0.0033` | `0.0013` | `0.0087` | `0.0225` |
| `0.1` | `0.00056` | `0.0` | `0.00139` | `0.00398` |

`R_pixel` 与 `R_feature` 的相关性：

| tau | Pearson correlation |
|---:|---:|
| `0.005` | `0.688` |
| `0.01` | `0.667` |
| `0.03` | `0.555` |
| `0.05` | `0.475` |
| `0.1` | `0.405` |

在主观察阈值 `tau=0.01` 下，平均 `R_feature=0.0205`，明显低于平均 `R_pixel=0.0611`。这表示页面局部变化没有扩散到大量 visual features。

## Model-Ready 核心结果

`model_ready_visual` 边界通过 `model.visual.merger` forward hook 捕获，输出已经是进入 LLM prefix 前的 visual embeddings。本轮结果验证了它和 patch-level feature 没有混淆：

| field | value |
|---|---:|
| feature boundary | `model_ready_visual` |
| feature source | `model.visual.merger` |
| feature dim | `5120` |
| feature token count | `377` 为主，少数 `392` |
| visual token grid | `[29,13]` 为主，少数 `[28,14]` |
| processor merge size | `2` |

同一批 100 个 pair 上，`R_pixel` 仍为 mean `0.0611`。model-ready visual tokens 的显著变化比例为：

| tau | mean `R_feature` | median | p90 | max | `R_feature > R_pixel` |
|---:|---:|---:|---:|---:|---:|
| `0.005` | `0.1547` | `0.1260` | `0.3156` | `0.6154` | `96 / 100` |
| `0.01` | `0.1171` | `0.0849` | `0.2395` | `0.4801` | `84 / 100` |
| `0.03` | `0.0764` | `0.0491` | `0.1671` | `0.3156` | `53 / 100` |
| `0.05` | `0.0602` | `0.0411` | `0.1385` | `0.2626` | `38 / 100` |
| `0.1` | `0.0376` | `0.0212` | `0.1016` | `0.1910` | `18 / 100` |

`R_pixel` 与 `R_feature` 的 Pearson correlation 在各阈值下约为 `0.68-0.70`，高于 patch-level 在高阈值下的相关性。这说明 model-ready boundary 中变化强度仍和 pixel 变化规模相关，但距离尺度被 merger/projector 明显放大。

关键解释：

- patch-level 的 `tau=0.01` 不能直接迁移到 model-ready embeddings。
- model-ready 下 `tau=0.05` 的平均 `R_feature=0.0602`，最接近平均 `R_pixel=0.0611`。
- model-ready 下 `tau=0.10` 更保守，平均 `R_feature=0.0376`，仍保持和 `R_pixel` 的相关性。

## Similarity 分段

按 page tile similarity 分段后的真实 pair 数和 patch-level `tau=0.01` 下结果：

| page similarity | pairs | mean `R_pixel` | mean `R_feature` | `R_feature / R_pixel` |
|---|---:|---:|---:|---:|
| `[0.85,0.90)` | 20 | `0.1230` | `0.0409` | `0.332` |
| `[0.90,0.95)` | 35 | `0.0705` | `0.0228` | `0.323` |
| `[0.95,0.98)` | 29 | `0.0339` | `0.0125` | `0.369` |
| `[0.98,0.99)` | 9 | `0.0156` | `0.00567` | `0.363` |
| `[0.99,1.00)` | 7 | `0.00781` | `0.00313` | `0.400` |

页面越相似，feature changed ratio 越低；比例上大致稳定在 `R_feature ~= 0.32-0.40 * R_pixel`。

model-ready visual tokens 在 `tau=0.05` 下与 pixel changed ratio 最匹配：

| page similarity | pairs | mean `R_pixel` | mean `R_feature` | `R_feature / R_pixel` |
|---|---:|---:|---:|---:|
| `[0.85,0.90)` | 20 | `0.1230` | `0.1261` | `1.025` |
| `[0.90,0.95)` | 35 | `0.0705` | `0.0667` | `0.946` |
| `[0.95,0.98)` | 29 | `0.0339` | `0.0343` | `1.009` |
| `[0.98,0.99)` | 9 | `0.0156` | `0.0153` | `0.981` |
| `[0.99,1.00)` | 7 | `0.00781` | `0.00493` | `0.631` |

因此如果后续以 model-ready visual embeddings 作为 cache/replacement 边界，`tau=0.05` 比 `tau=0.01` 更适合作为主观察阈值候选。

## 空间局部性

patch-level changed pixel patch 对应的 feature 差异明显高于 unchanged 区域：

| region | mean cosine distance |
|---|---:|
| changed pixel patch tokens | `0.002154` |
| unchanged pixel patch tokens | `0.000753` |

changed / unchanged 均值比约 `2.86x`。在 `100` 个 pair 中，有 `79` 个 pair 的 changed 区域 feature mean 高于 unchanged 区域。

距离 changed region 的聚合曲线也支持局部性：

| token Manhattan distance | mean cosine distance |
|---:|---:|
| `0` | `0.002154` |
| `1` | `0.001480` |
| `2` | `0.001526` |
| `3` | `0.000943` |
| `7` | `0.000788` |
| far tail | mostly `1e-5` to `1e-4` |

曲线有局部波动，但总体是近 changed region 更高，远处逐步下降。

model-ready visual tokens 的空间局部性更明显，但背景扰动也更高：

| region | mean cosine distance |
|---|---:|
| changed pixel region tokens | `0.04887` |
| unchanged pixel region tokens | `0.00738` |

changed / unchanged 均值比约 `6.62x`，且 `100 / 100` 个 pair 的 changed 区域 feature mean 高于 unchanged 区域。

距离 changed region 的 model-ready 聚合曲线：

| token Manhattan distance | mean cosine distance |
|---:|---:|
| `0` | `0.04887` |
| `1` | `0.01737` |
| `2` | `0.00845` |
| `3` | `0.00722` |
| `10` | `0.00427` |
| `20` | `0.00288` |
| `25` | `0.000898` |
| `30` | `0.000252` |

这说明 merger/projector 后仍保留明显局部性；但因为每个 token 已合并多个 patch 并投影到 LLM hidden size，非 changed 区域的弱变化比 patch-level 更高。

## 扩散风险

patch-level 统计 `R_feature > R_pixel` 的 pair 数：

| tau | count |
|---:|---:|
| `0.005` | `11 / 100` |
| `0.01` | `2 / 100` |
| `0.03` | `0 / 100` |
| `0.05` | `0 / 100` |
| `0.1` | `0 / 100` |

`tau=0.005` 较敏感，会把弱扰动也计入显著变化。`tau=0.01` 下只有 2 个 pair 出现 `R_feature > R_pixel`；`tau>=0.03` 下没有出现。

model-ready boundary 的扩散风险更依赖阈值：

| tau | count |
|---:|---:|
| `0.005` | `96 / 100` |
| `0.01` | `84 / 100` |
| `0.03` | `53 / 100` |
| `0.05` | `38 / 100` |
| `0.1` | `18 / 100` |

这不表示 model-ready embeddings 完全失去局部性，而是说明 merger/projector 后 cosine distance 的尺度明显不同。判断 model-ready cache 可行性时，应重新标定阈值，不能沿用 patch-level 的 `tau=0.01`。

## 结论

当前 100-pair 结果支持继续研究 Patch Feature Cache / Model-ready Visual Feature Cache，但两层边界的结论不同：

- similar GUI pages 的局部 pixel changes 在 `model.visual` final feature space 中基本保持局部。
- patch-level `R_feature` 通常小于 `R_pixel`，没有观察到大面积 feature 扩散。
- model-ready `model.visual.merger` 输出中仍有清晰局部性，changed 区域距离显著高于 unchanged 区域，且距离 changed region 越远总体越低。
- model-ready embeddings 的距离尺度更大，`tau=0.01` 过敏；`tau=0.05` 更接近 pixel changed ratio，`tau=0.10` 更保守。

推荐阈值口径：

- patch-level `model.visual`：继续把 `tau=0.01` 作为主观察阈值之一。
- model-ready `model.visual.merger`：优先观察 `tau=0.05`，同时报告 `tau=0.10` 作为保守口径。

## 限制

- 当前已分析 `model.visual` final patch features 和 `model.visual.merger` model-ready visual embeddings，但尚未实现真实 feature cache。
- model-ready locality 只证明进入 LLM prefix 前的 visual embeddings 仍具空间局部性；它不等价于 full-prefix KV cache。KV 已经过 LLM self-attention 混合，near/patch 页面不应直接局部替换 KV。
- `app` 字段为空，无法做有效 app 维度分析。
- 当前尚未比较 intermediate layers，因此还不能判断 feature locality 随 encoder depth 的变化。

## 下一步

优先补充：

1. 对 model-ready boundary 补跑更细阈值：`0.02`, `0.03`, `0.04`, `0.05`, `0.075`, `0.10`, `0.15`，确认 `tau=0.05` 附近是否稳定。
2. 如果模型接口支持 hidden states，比较 intermediate layers 与 final layer，观察 feature locality 是否随 depth 变弱。
3. 如果继续做真实 cache，应优先验证 exact prefix KV cache；near/patch 页面仍应从 visual feature / model-ready visual embedding replacement 起步，不直接复用 KV。
