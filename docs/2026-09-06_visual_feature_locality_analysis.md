# Visual Feature Locality Analysis

日期：2026-09-06

## 目标

本实验用于回答：

> 当 GUI 页面只有少量 patch 发生变化时，vision encoder 输出中是否也只有少量对应 feature 发生显著变化？

实验只做 profiling / analysis，不实现真正的 patch feature cache，也不训练或修改模型权重。后续 feature cache 可基于这里预留的边界继续实现：

```text
cached feature + changed patch recomputation + feature replacement
```

## 实验配置

主要结果目录：

```text
results/feature_locality_analysis/androidcontrol_1000_broad_100
```

运行配置：

- 模型：`/data2/home/models/Qwen3.8-27B`
- 数据：`data/androidcontrol_1000/test.json`
- visual token mode：`aggressive_reduce`
- feature source：`model.visual`
- feature layer：`final`
- feature metric：`cosine_distance`
- threshold：`0.005`, `0.01`, `0.03`, `0.05`, `0.1`
- page similarity：tile-based
- pair scope：dataset
- max pairs：100
- tile grid：`8 x 16`

本次结果中：

- valid pair-layer rows：`100`
- error count：`0`
- hit type：`patch_candidate=55`, `near=45`
- visual feature tokens：多数为 `1508`，少数为 `1568`

## 核心结果

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

## Similarity 分段

按 page tile similarity 分段后的真实 pair 数和 `tau=0.01` 下结果：

| page similarity | pairs | mean `R_pixel` | mean `R_feature` | `R_feature / R_pixel` |
|---|---:|---:|---:|---:|
| `[0.85,0.90)` | 20 | `0.1230` | `0.0409` | `0.332` |
| `[0.90,0.95)` | 35 | `0.0705` | `0.0228` | `0.323` |
| `[0.95,0.98)` | 29 | `0.0339` | `0.0125` | `0.369` |
| `[0.98,0.99)` | 9 | `0.0156` | `0.00567` | `0.363` |
| `[0.99,1.00)` | 7 | `0.00781` | `0.00313` | `0.400` |

页面越相似，feature changed ratio 越低；比例上大致稳定在 `R_feature ~= 0.32-0.40 * R_pixel`。

## 空间局部性

changed pixel patch 对应的 feature 差异明显高于 unchanged 区域：

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

## 扩散风险

统计 `R_feature > R_pixel` 的 pair 数：

| tau | count |
|---:|---:|
| `0.005` | `11 / 100` |
| `0.01` | `2 / 100` |
| `0.03` | `0 / 100` |
| `0.05` | `0 / 100` |
| `0.1` | `0 / 100` |

`tau=0.005` 较敏感，会把弱扰动也计入显著变化。`tau=0.01` 下只有 2 个 pair 出现 `R_feature > R_pixel`；`tau>=0.03` 下没有出现。

## 结论

当前 100-pair 结果支持 Patch Feature Cache 的可行性：

- similar GUI pages 的局部 pixel changes 在 `model.visual` final feature space 中基本保持局部。
- `R_feature` 通常小于 `R_pixel`，没有观察到大面积 feature 扩散。
- changed pixel region 的 feature distance 显著高于 unchanged region。
- feature distance 随离 changed region 的空间距离总体下降。

推荐把 `tau=0.01` 作为后续主观察阈值之一：它能过滤大量微弱噪声，同时仍保留较明显的 feature change。

## 限制

- 当前只分析 `model.visual` 的 `final` 输出。
- 当前 `feature_token_count` 多数为 `1508`，而 metadata 中估算的 merged visual token count 为 `377`；因此结果更接近 vision tower patch-level feature locality，不完全等价于 projector / merger 后的 model-ready visual embeddings。
- `app` 字段为空，无法做有效 app 维度分析。
- 当前尚未比较 intermediate layers，因此还不能判断 feature locality 随 encoder depth 的变化。

## 下一步

优先补充两类实验：

1. 提取 merger / projector 后的 model-ready visual embeddings，确认进入 LLM prefix 前的 `377` 个 visual tokens 是否同样局部。
2. 如果模型接口支持 hidden states，比较 intermediate layers 与 final layer，观察 feature locality 是否随 depth 变弱。

