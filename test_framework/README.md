# 华为手机 GUI 批量测评

项目通过视觉语言模型生成手机操作 action，经 ADB 控制一台真实华为 Android 手机，并保存逐步截图、动作、TTFT、模型生成时长、端到端时长和任务总时长。

完整流程由三部分组成：本机连接手机并建立反向 SSH 通道；Jupiter 使用两张 GPU 部署 Qwen3.8；服务器运行 `benchmark_runner.py`，通过反向通道操作本机手机。

## 1. 本机连接手机

开启手机 USB 调试、解锁并授权后确认设备：

```bash
adb devices -l
```

建立 Jupiter 到本机 ADB 的自动通道：

```bash
scripts/start_adb_bridge.sh Jupiter
```

脚本会建立 `Jupiter localhost:2222 -> 本机 22` 的反向 SSH 通道，并在服务器实际检查手机连接。ADB transport 默认使用 `auto`：本机存在 USB 设备时直连，否则经 SSH 通道执行。

## 2. 服务器启动模型

在 Jupiter 的项目目录执行：

```bash
cd /data2/home/luyijie/test_framework
CUDA_VISIBLE_DEVICES=4,5 scripts/start_qwen38_server.sh
```

默认模型路径为 `/data2/home/luyijie/models/Qwen3.8-27B`，服务地址为 `http://127.0.0.1:8018/v1`，tensor parallel size 为 2。

## 3. 运行手机任务

在 Jupiter 另一个终端执行：

```bash
cd /data2/home/luyijie/test_framework
python3 benchmark_runner.py \
  --tasks GUI操控dev测评集.csv \
  --output outputs/qwen38_gui_dev_latest \
  --base-url http://127.0.0.1:8018/v1 \
  --model /data2/home/luyijie/models/Qwen3.8-27B \
  --max-steps 12
```

手机 action prompt 统一维护在 `phone_prompt.py`。支持 `tap`、`swipe`、`type`、`back`、`home`、`wait`、`complete` 和 `impossible`，坐标使用整屏 `0-1000` 归一化整数。

主要输出包括：

- 每个任务目录中的 `metadata.json`、`steps.jsonl`、逐步截图、`final.png` 和 `done.json`
- `latency.csv`：逐步 TTFT、模型总时长、请求 E2E 和步骤 E2E
- `task_timing.csv`：任务总时长
- `summary.jsonl`：每项任务的终止状态

如进程中断，可先清理未完成任务再从指定任务继续：

```bash
python3 scripts/prepare_benchmark_resume.py outputs/qwen38_gui_dev_latest
python3 benchmark_runner.py \
  --tasks GUI操控dev测评集.csv \
  --output outputs/qwen38_gui_dev_latest \
  --start-task 20
```

## 人工审阅前端

本地执行：

```bash
scripts/start_review_console.sh
```

默认打开 `http://127.0.0.1:8765`。审阅结果独立写入 `outputs/qwen38_gui_dev_annotated_full_v2_20260821/review_annotations.json`，不会修改原始任务、截图和时延文件。

