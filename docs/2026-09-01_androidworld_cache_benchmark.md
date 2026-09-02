# 2026-09-01 AndroidWorld Cache Benchmark

本仓库新增了一个小规模 AndroidWorld cache benchmark adapter。目标不是复现完整 AndroidWorld leaderboard，而是复用 AndroidWorld 官方 task registry / suite / live emulator environment，用动态参数化任务构造 cache-oriented warm-up/evaluation split。

参考官方 AndroidWorld 仓库：

- https://github.com/google-research/android_world
- 官方 README 说明 AndroidWorld 使用 live Android emulator，包含 116 个手写任务、20 个 app，并通过随机参数生成大量任务变体。
- 官方 `run.py` 通过 `registry.TaskRegistry()`、`suite_utils.create_suite(..., n_task_combinations, seed, tasks)` 和 `env_launcher.load_and_setup_env(...)` 运行 benchmark。

## 新增入口

- `scripts/run_androidworld_cache_benchmark.py`
- `scripts/summarize_androidworld_cache.py`
- `configs/androidworld_cache_subset.json`

runner 只在 adapter 层接入 AndroidWorld，不复制 AndroidWorld task 逻辑。模型推理继续使用本项目 `hf_gui_baseline.profile_infer_one()`，cache 继续使用 `PageLevelCache` 的 `off|observe|inputs`、`exact|dhash|tile` 配置。

## 默认 Subset

默认任务列表在 `configs/androidworld_cache_subset.json` 中：

```text
ContactsAddContact
ClockTimerEntry
ExpenseAddSingle
MarkorCreateFolder
MarkorCreateNote
SimpleSmsSend
SimpleSmsReply
SimpleCalendarAddOneEventTomorrow
SimpleCalendarAddOneEventRelativeDay
SimpleCalendarDeleteOneEvent
```

默认 `n_task_combinations=5`，即 10 个 task template x 5 个参数实例 = 50 episodes。远端资源紧张时使用 `--limit_episodes` 或 `--n_task_combinations 1` 缩小。

## Warm-up / Evaluation

默认：

- warm-up seed: `30`
- evaluation seed: `31`

两者 task template 相同，但 AndroidWorld 官方 suite 会用不同随机种子生成不同联系人、短信、时间、文件名等参数实例。cache evaluation 可通过 `--cache_input` 加载 warm-up 导出的页面指纹索引，用于观测跨 instance page/cache reuse。

当前 cache 仍沿用本项目 page-level baseline：

- `off`: baseline，不做 cache lookup。
- `observe`: 记录 exact/near/patch candidate，不复用模型输入。
- `inputs`: exact processor input cache；跨进程只持久化页面指纹，不持久化 processor tensor。

## AndroidWorld 外部依赖

远端需要先完成 AndroidWorld 官方环境：

```bash
git clone https://github.com/google-research/android_world.git
cd android_world
pip install -r requirements.txt
python setup.py install
```

Android emulator 推荐按官方设置使用 Pixel 6、API 33，并从命令行启动，带 `-grpc 8554`：

```bash
EMULATOR_NAME=AndroidWorldAvd
~/Android/Sdk/emulator/emulator -avd "${EMULATOR_NAME}" -no-snapshot -grpc 8554
```

首次安装 AndroidWorld app/权限时加 `--perform_emulator_setup`，后续正常评测不要重复加。

## Pip 依赖冲突处理

AndroidWorld 官方 `requirements.txt` 当前 pin 了：

```text
protobuf==5.29.5
numpy==1.26.3
```

这与 `tensorflow-cpu 2.21.0` 不兼容，因为后者要求 `protobuf>=6.31.1,<8.0.0`，并且常见的 `ml-dtypes 0.6.0` 要求 `numpy>=2.0.0`。因此不要在同一个 env 同时安装 AndroidWorld 和本项目仅用于 AndroidControl TFRecord 预处理的 `requirements-androidcontrol.txt`。

推荐为 AndroidWorld 单独建环境：

```bash
conda create -n android_world_cache python=3.11 -y
conda activate android_world_cache
cd /path/to/android_world
pip install -r requirements.txt
python setup.py install
cd /path/to/GUI-Agent-worksapce
pip install -r requirements.txt
```

如果已经在当前 env 里装出了冲突，且这个 env 只用于 AndroidWorld benchmark，可以移除 TensorFlow 相关包后重装 AndroidWorld 依赖：

```bash
pip uninstall -y tensorflow tensorflow-cpu ml-dtypes
cd /path/to/android_world
pip install -r requirements.txt
python setup.py install
```

如果仍需要运行 `scripts/prepare_androidcontrol.py`，请在另一个专门的 AndroidControl preprocessing env 里安装 `requirements-androidcontrol.txt`，不要复用 AndroidWorld benchmark env。

## 推荐远端执行顺序

1. AndroidWorld 环境 smoke test，默认用 mock `complete` 动作，不加载本地大模型：

```bash
python scripts/run_androidworld_cache_benchmark.py \
  --android_world_path /path/to/android_world \
  --run_mode smoke \
  --output results/androidworld_cache/smoke \
  --perform_emulator_setup \
  --limit_episodes 1
```

2. baseline 小规模测试，cache disabled：

```bash
CUDA_VISIBLE_DEVICES=4,5 python scripts/run_androidworld_cache_benchmark.py \
  --android_world_path /path/to/android_world \
  --run_mode baseline \
  --output results/androidworld_cache/baseline_smoke \
  --model_path /data2/home/models/Qwen3.8-27B \
  --n_task_combinations 1 \
  --limit_episodes 3 \
  --page_cache_mode off
```

3. warm-up cache：

```bash
CUDA_VISIBLE_DEVICES=4,5 python scripts/run_androidworld_cache_benchmark.py \
  --android_world_path /path/to/android_world \
  --run_mode warmup \
  --output results/androidworld_cache/warmup \
  --model_path /data2/home/models/Qwen3.8-27B \
  --page_cache_mode observe \
  --page_cache_scope dataset \
  --page_cache_similarity tile
```

4. evaluation，加载 warm-up 页面 cache：

```bash
CUDA_VISIBLE_DEVICES=4,5 python scripts/run_androidworld_cache_benchmark.py \
  --android_world_path /path/to/android_world \
  --run_mode evaluation \
  --output results/androidworld_cache/eval_cache \
  --model_path /data2/home/models/Qwen3.8-27B \
  --page_cache_mode observe \
  --page_cache_scope dataset \
  --page_cache_similarity tile \
  --cache_input results/androidworld_cache/warmup/page_cache_records.jsonl
```

5. baseline vs cache 汇总：

```bash
python scripts/summarize_androidworld_cache.py \
  --baseline results/androidworld_cache/baseline_smoke \
  --cache results/androidworld_cache/eval_cache \
  --output results/androidworld_cache/comparison_summary.json
```

## 输出数据

runner 输出：

- `run_config.json`: 完整运行参数。
- `steps.jsonl`: step-level trajectory 记录。
- `episodes.jsonl`: episode-level reward/success/latency。
- `summary.json`: success rate、cache hit rate、latency、model invocation count、cache lookup overhead、按 task/app/cache reuse group 汇总。
- `page_cache_records.jsonl`: 可导入下一次 evaluation 的页面指纹 records。
- `episodes/<episode_id>/step_*.png`: 每一步 screenshot。

每个 step 至少记录：

- task template、task params/seed、task goal、episode_id、step_id。
- screenshot、UI elements/accessibility summary。
- previous/current action、AndroidWorld JSONAction。
- model input/output、input/output tokens、inference latency、step latency。
- cache lookup latency、hit/miss、hit entry、similarity score、fallback/model_invoked。
- episode final reward/success。

## 当前限制

- 本地 Windows 环境没有 AndroidWorld emulator、模型和 GPU，因此只做静态检查；真实 task 初始化、动作执行、reward 需要远端 emulator 验证。
- 跨进程 warm-up/evaluation 目前持久化的是 page fingerprint index；processor input tensor cache 仍是进程内 exact cache。
- adapter 使用本项目归一化坐标 action contract，再转换到 AndroidWorld `JSONAction`。复杂输入场景可能需要后续增加 index-based action prompt，但第一版先保持与现有推理接口一致。
