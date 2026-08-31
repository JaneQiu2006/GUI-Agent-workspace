# 2026-08-31 GUI Agent 缓存式推理加速设计

本文基于当前 `hf_gui_baseline.py` / `eval_androidcontrol.py` / `profile_androidcontrol.py` 静态推理链路设计缓存方案，不训练模型，不重搭已有推理与评测功能。当前已验证的单请求推荐基线是：

```text
max_new_tokens=48
visual_token_mode=aggressive_reduce
batch_size=1
dtype=auto
attn_implementation=None
```

缓存方案应先作为可选插桩接入这条基线，所有实验继续报告 `strict`、`gui_only`、`by_gt_type`、输出健康度和显存。

## 结论优先

推荐采用三级层次化 cache，但分阶段实现：

1. **Page-level baseline**：先实现无模型页面指纹分析、processor input cache、精确 full-prefix KV cache 小样本验证。不要一开始做模糊 KV 复用。
2. **Patch-level extension**：页面大部分不变时只把 tile diff、changed bbox 和相似度记录下来；第一版不复用局部 KV，只用于 gating 和风险分析。
3. **Semantic cache**：第二阶段再做，优先用轻量规则和已有信息抽象 UI/text/layout/task-relevant state，不引入额外重模型。

最现实的第一步收益来自 **full multimodal prefix/KV 精确命中**。但它的 key 必须包含完整模型输入，因此只有同一 screenshot、同一 prompt、同一视觉 token 配置和同一模型版本完全一致时才能安全复用。跨任务或页面小变动的复用应先只做统计和 ablation，不能直接跳过 prefill。

## 现有数据流

当前静态推理路径在 `test_framework/hf_gui_baseline.py` 中已经拆分得较清楚：

```text
sample(image_path, task, action_hint)
  -> build_gui_messages()
  -> apply_chat_template_without_thinking()
  -> process_vision_info()
  -> processor(...)
  -> inputs.to(device)
  -> model.generate(...)
  -> processor.batch_decode(...)
  -> parse_action / canonicalize_action
  -> eval/profile JSON
```

当前 profiling 字段只把 `model.generate()` 作为一个整体记录为 `generate_seconds`，还没有拆出 `prefill`、`TTFT`、逐 token decode。缓存实验前应先补这类观测，否则无法判断 page/prefix cache 是否命中了真正瓶颈。

## 推荐整体架构

新增一个可选的 cache 层，放在 `hf_gui_baseline.py` 外侧或以轻量参数注入，避免污染默认 baseline：

```text
CacheAwareInferencer
  PageFingerprintIndex
  ProcessorInputCache
  PrefixKVCache
  SemanticStateCache  # 第二阶段，仅定义接口，不立即完整实现
```

推荐数据流：

```text
1. 构造 messages 和 chat_text
2. 计算 page fingerprint
3. 查询 page index，得到 exact/similar/patch-level 候选
4. 查询 processor input cache
5. 查询 full-prefix KV cache
6. cache miss 时走原始 preprocess + generate
7. 写回 fingerprint、processor inputs、可选 prefix KV、profiling 记录
8. decode 和 action parser 保持原逻辑
```

默认开关建议：

```text
--cache_mode off | inputs | exact_kv | observe
--cache_scope trajectory | dataset | session
--page_similarity exact | dhash | tile
--cache_max_entries N
--cache_max_gpu_gb X
```

其中 `observe` 只统计潜在 hit，不复用任何值，用于先评估收益上限和 accuracy 风险。

## 各级 Cache 设计

| 层级 | 输入 | Key | Value | 命中条件 | 失效条件 | 第一阶段建议 |
| --- | --- | --- | --- | --- | --- | --- |
| Raw screenshot cache | `image_path` / bytes | `sha256(image_bytes)` | PIL/RGB image、尺寸、bytes hash | bytes 完全一致 | 文件变化、尺寸变化 | 可做，但收益很小 |
| Page fingerprint index | screenshot | `sha256` + `dhash/phash` + tile hashes | 页面元信息、相似候选、diff tiles | exact hash 或相似度过阈值 | 截图尺寸/方向变、状态栏时间等动态区域过多 | 必做 observe |
| Processor input cache | messages、chat_text、image | `model_id + processor_id + visual_token_mode + chat_text_hash + image_hash` | CPU tensors：`input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw` 等 | 完整 key 一致 | prompt/processor/视觉 token 参数变化 | MVP 可实现，低风险 |
| Vision output cache | image tensors | `model_id + vision_config + visual_token_mode + image_hash` | vision encoder hidden states / visual embeds | 图像与视觉配置一致 | 模型权重、视觉 encoder、resize 策略变化 | 暂不优先，需碰模型私有接口 |
| Full multimodal prefix KV cache | 完整 `inputs` | `model_id + dtype + attn + device_map + input_ids_hash + pixel_values_hash + image_grid_hash` | `past_key_values`、prefix length、attention/position 元信息 | 完整输入 prefix 一致 | 任意 token、图片、位置编码、dtype、device 变化 | 第一阶段重点验证 |
| Patch-level cache | screenshot + previous page | stable tile hashes + changed tile mask | unchanged regions 元信息、changed bbox | 大部分 tile 一致且改动区域非任务关键 | 关键控件/目标区域变化、键盘/弹窗/滚动内容变化 | 只观察，不复用 KV |
| Semantic cache | page fingerprint + optional UI/text/layout state + task | `semantic_state_hash + task_intent_hash` | UI element graph、OCR/text、layout slots、task-relevant summary | 语义状态一致或近似一致 | 元素可用性、选中态、输入框内容、弹窗变化 | 第二阶段 |

## Cache Key 细节

精确 key 必须包含：

- `model_path` 或模型权重 hash / revision。
- `processor` 版本、`transformers` 版本、`qwen-vl-utils` 版本。
- `dtype`、`attn_implementation`、`device_map`、视觉 token 参数。
- `chat_text` hash，不能只用 task hash。
- `image_bytes_sha256`、尺寸、通道格式。
- `input_ids` hash、`attention_mask` hash、`pixel_values` hash、`image_grid_thw` hash。
- prompt/action contract 版本，可用 `PHONE_SYSTEM_PROMPT` hash。

模糊页面 key 不应直接用于 KV 复用，只用于候选召回：

```text
page_fingerprint = {
  sha256,
  dhash64,
  phash64,
  size,
  tile_hashes: 8x16 或 10x20,
  ignored_regions: status_bar/nav_bar 可选,
}
```

推荐阈值从保守开始：

- exact hit：`sha256` 完全一致。
- near hit：`dhash_hamming <= 4` 且 tile unchanged ratio >= 0.98。
- patch hit candidate：tile unchanged ratio >= 0.90，且 changed bbox 不覆盖目标区域估计。

第一阶段只把 near/patch 作为日志字段，不用来复用 full KV。

## 页面小 patch 变化的处理

GUI trajectory 中常见变化包括时间、电量、loading spinner、输入框光标、按钮选中态、软键盘、toast、弹窗、列表滚动少量偏移。建议分三类处理：

1. **动态噪声区**：状态栏时间、电量、导航栏、光标闪烁。fingerprint 可忽略或降权。
2. **局部 UI 状态变化**：checkbox、tab selected、按钮 enabled/disabled、输入框内容。会影响动作选择，只能作为 patch candidate，不直接复用动作或 KV。
3. **结构变化**：弹窗、键盘、页面跳转、列表滚动。必须 miss，除非 semantic cache 能证明任务相关状态不变。

MVP 的关键原则：可以复用 exact processor inputs / exact KV；相似页面只统计，不执行近似复用。

## Semantic Cache 设计

Semantic cache 的目标是允许视觉上有变化，但任务相关语义状态基本不变时复用中间 representation。侧端轻量化场景下不建议依赖额外大模型。可选信号按优先级：

1. Android / HarmonyOS 可访问树：`uiautomator dump` 或系统 accessibility tree，如果在线设备评测可用。
2. 轻量 OCR：只在侧端已有 OCR 能力时使用；否则不要为了 cache 引入重依赖。
3. 图像规则：connected components、模板级 text bbox、控件轮廓、颜色块和布局 slot。
4. 模型已有中间输出：vision encoder pooled / selected visual token summary，但这需要模型接口改造。

语义状态建议结构：

```text
SemanticState {
  app_or_screen_id
  visible_text_hashes
  ui_elements: [{type, text_hash, bbox, enabled, selected}]
  layout_grid_hash
  modal_keyboard_status
  task_relevant_slots
}
```

Semantic key：

```text
semantic_key = hash(
  screen_id,
  normalized_visible_text,
  normalized_element_graph,
  keyboard/modal flags,
  task_intent_class
)
```

Semantic cache 的 value 不建议第一版存最终 action。更稳妥的是存：

- 页面/元素候选集合。
- task-relevant element bbox。
- 上一次 visual token policy 决策。
- 是否允许走更低视觉 token 或复用 page-level candidate。

## Hierarchical Cache 组合

推荐查询顺序：

```text
Exact full-prefix KV cache
  -> exact processor input cache
  -> exact page/vision cache
  -> near page patch candidate
  -> semantic state candidate
  -> normal inference
```

安全策略：

- exact KV hit：允许跳过 prefill，直接 decode。
- exact processor input hit：跳过 `process_vision_info()` 和 `processor(...)`。
- near page hit：只记录 `near_hit=true`，不跳过模型。
- semantic hit：第一版只影响视觉 token policy 或候选日志，不直接复用 action。
- action cache：只允许在完整 `chat_text + image + model config` 完全一致时作为 debug upper bound，不进入默认方案。

## 最适合当前项目的 MVP

MVP 分三步，尽量少侵入现有推理流程。

### Step 1：Cache Potential Analyzer

新增脚本建议：

```text
scripts/analyze_page_cache_potential.py
```

读取 `data/androidcontrol_mini/test.json` 或真实 trajectory 输出目录，计算：

- exact screenshot duplicate rate。
- same trajectory 内 near-page rate。
- tile unchanged ratio 分布。
- changed bbox 面积占比。
- 按 GT action type 的重复/相似分布。
- 如果有 history，统计相同页面但不同 `task/current step` 的比例。

输出到：

```text
results/cache_analysis/page_potential.json
```

这一步不加载模型，能先回答“页面重复现象是否足够多”。

### Step 2：Processor Input Cache

新增模块建议：

```text
test_framework/cache_utils.py
test_framework/cache_inference.py
```

最小接口：

```python
class InferenceCache:
    def get_processor_inputs(self, key): ...
    def put_processor_inputs(self, key, value): ...
    def get_prefix_kv(self, key): ...
    def put_prefix_kv(self, key, value): ...
```

接入点优先放在 `preprocess_inputs()` / `preprocess_batch_inputs()` 附近，或者新增 `cache_preprocess_inputs()` wrapper，不改默认函数行为。

收益预期：只节省约 0.1-0.2s 的 Python/processor 输入侧开销，不是最终目标，但准确率风险最低，可以验证 key、hit/miss、日志和评测集成。

### Step 3：Exact Full-Prefix KV Cache Prototype

新增一个实验性 generate wrapper：

```text
generate_response_with_prefix_cache(...)
```

思路：

1. 对完整 inputs 做一次 forward prefill，保存 `past_key_values`。
2. 同一完整 prefix 命中时，从最后一个输入 token 开始 greedy decode。
3. 保持 `do_sample=False`、`max_new_tokens=48`、同样 eos/pad 设置。
4. 输出必须与原 `model.generate()` 在 exact key 下逐样本一致或 action 一致。

注意：Transformers 多模态模型的 `past_key_values`、position ids、cache position 在不同版本中可能有私有细节。这个 prototype 应隔离在新模块，并保留 `--cache_mode off` 回退。

## 需要新增/修改的模块和接口

建议新增：

- `test_framework/cache_fingerprint.py`：截图 hash、dhash/phash、tile hash、diff bbox。
- `test_framework/cache_store.py`：LRU / TTL / GPU memory budget 管理。
- `test_framework/cache_inference.py`：cache-aware preprocess/generate wrapper。
- `scripts/analyze_page_cache_potential.py`：无模型页面重复潜力分析。
- `scripts/profile_cache_androidcontrol.py`：缓存版 profiling，或给现有 `profile_androidcontrol.py` 增加可选参数。

建议最小修改：

- `hf_gui_baseline.py`：不改默认行为；只新增可选 wrapper 或参数入口。
- `profile_androidcontrol.py`：增加 cache 配置和 cache metrics 汇总。
- `eval_androidcontrol.py`：只在需要做完整准确率回归时传 cache 参数。
- `run_accel_experiments.py`：后续再加入 E21+ 缓存实验，不在 MVP 初期强行扩展矩阵。

## Profiling 指标

保留现有字段，并新增：

- `cache_mode`
- `page_cache_hit`
- `page_cache_hit_type`: `exact` / `near` / `patch_candidate` / `miss`
- `processor_cache_hit`
- `prefix_kv_cache_hit`
- `semantic_cache_hit`
- `similarity_dhash_hamming`
- `tile_unchanged_ratio`
- `changed_bbox_area_ratio`
- `cache_lookup_seconds`
- `cache_write_seconds`
- `prefill_seconds`
- `ttft_seconds`
- `decode_seconds_model`
- `decode_tokens_per_second`
- `generate_seconds`
- `total_seconds`
- `input_tokens`
- `visual_tokens` 或 `image_grid_thw`
- `kv_cache_bytes_gpu`
- `kv_cache_bytes_cpu`
- `cache_entries`
- `cache_evictions`
- `peak_gpu_memory_gb`

其中 `prefill_seconds` 和 `ttft_seconds` 是判断 KV cache 是否有效的核心指标。

## 实验设计

由简单到复杂：

1. **C00 observe only**：不复用，统计 exact/near/patch hit 潜力。
2. **C01 processor input cache exact**：只缓存 processor outputs，验证 zero accuracy regression 和输入侧节省。
3. **C02 full-prefix KV exact single repeat**：单张图同一 prompt 重复 N 次，要求 raw response 或 canonical action 一致。
4. **C03 full-prefix KV exact trajectory/dataset**：在 AndroidControl mini 上启用 exact KV；如果 hit 低，也要报告潜在上限。
5. **C04 cache + E11**：固定 `aggressive_reduce + max_new_tokens=48`，确认与当前最好基线可叠加。
6. **C05 cache + batch_size=2/4**：只在 C04 无准确率回退后测试吞吐，观察显存压力。
7. **C06 near-page observe ablation**：只记录 near/patch，如果强行复用上一页 action 会导致多少错误，用作风险上界。
8. **C07 semantic observe**：用轻量 UI/text/layout state 统计 semantic hit，不直接复用 KV 或 action。

验收标准沿用现有加速实验：

- `gui_only.step_success_rate >= 91.89%`
- `gui_only.type_accuracy >= 97.30%`
- `pred_unknown == 0`
- `hit_max_new_tokens == 0`
- CLICK step success 不明显下降
- 单请求延迟或吞吐有足够收益
- 显存不超过部署预算

## Accuracy 风险与 Ablation

| 风险 | 表现 | 对应 ablation |
| --- | --- | --- |
| 相似页面误命中 | 点击旧控件、忽略弹窗/键盘/选中态 | exact-only vs near-observe vs forced-near upper-risk |
| task/current step 不同 | 同一页面上下一步动作不同 | key 中加入完整 `chat_text`；对 action cache 做禁用对照 |
| 局部状态变化 | checkbox/tab/input 内容变化导致动作变 | changed bbox 与预测点重叠分析 |
| 视觉 token policy 改变 | cache value 与 `aggressive_reduce/default` 不兼容 | key 中加入 resolved visual token mode |
| 模型/processor 版本变化 | KV 或 pixel_values 不可复用 | key 中加入版本与 prompt hash |
| KV 位置编码错误 | raw response 漂移或解析失败 | exact repeat raw/action equivalence test |
| 显存膨胀 | cache 抵消延迟收益或 OOM | GPU KV budget、CPU offload、LRU ablation |
| batch + cache 交互 | left padding、position ids、trim decode 出错 | batch=1/2/4 分开验证，保留输出健康度 |

## 第一周可执行计划

1. 新增 `analyze_page_cache_potential.py`，先在 AndroidControl mini 和已有真实 trajectory 输出上统计 exact/near/patch hit。
2. 给 profiling 输出补 cache 相关字段，但默认 `cache_mode=off`。
3. 实现 processor input cache exact 模式，跑 `--limit 5` profile，确认结果完全一致。
4. 写一个 isolated exact KV repeat prototype，只在单图重复输入上验证 raw/action 一致性和 `prefill_seconds` 降低。
5. 若 C02 成立，再接入 AndroidControl mini 完整 eval；否则停止在 processor/page observe 层，重新评估是否值得改 generate。

第一阶段不实现完整 semantic cache，也不做近似 KV 复用。semantic cache 先以 observe 指标和接口定义存在，等 page-level exact cache 的收益和风险明确后再推进。
