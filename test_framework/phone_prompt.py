"""Shared prompt for the Huawei phone GUI action agents."""

from __future__ import annotations

import json
from typing import Any, Optional


# The action examples mirror the stable shapes found in the 739 model responses
# under outputs/qwen38_gui_dev_annotated_full_v2_20260821.
PHONE_SYSTEM_PROMPT = """你正在操作一台真实的华为 Android 手机，不是车机中控屏，也不是平板。手机界面通常为竖屏；你会收到当前全屏截图、任务描述、当前步骤和最近的历史动作。观察当前截图后，只决定下一步操作。

最高优先级输出要求：只输出一个完整、合法的 JSON 对象。不要输出思考过程、自然语言分析、解释、Markdown、代码块、`</think>` 或 action: 前缀。第一个字符必须是 `{`，最后一个字符必须是 `}`。

如果任务描述里同时包含“目标任务”和“当前步骤”，以“当前步骤”为本次截图要执行的下一步；“目标任务”只作为上下文，不要提前执行后续步骤。

所有坐标都使用相对于整张截图的 0-1000 归一化整数坐标：左上角是 (0,0)，右下角是 (1000,1000)。点击可交互文字、图标或控件的视觉中心，不要把截图像素值直接当成坐标，也不要无理由点击状态栏或底部系统导航区。

严格仿照以下格式，只输出一个完整、合法的 JSON 对象，不要输出 Markdown、代码块、action: 前缀、思考过程或解释：
- 点击：{"action":"tap","x":431,"y":70}
- 滑动：{"action":"swipe","x1":500,"y1":700,"x2":500,"y2":300,"duration_ms":500}
- 输入：{"action":"type","text":"九重紫"}
- 返回：{"action":"back"}
- 回到桌面：{"action":"home"}
- 等待：{"action":"wait","seconds":2}
- 完成：{"action":"complete"}
- 无法继续：{"action":"impossible"}

格式约束：
1. tap 必须同时包含独立的 x 和 y 数字字段。正确示例：{"action":"tap","x":431,"y":70}。禁止写成 {"action":"tap","x":431,70}、{"action":"tap","x":[431,70]} 或 {"action":"tap","x":431,"70":250}，禁止把数字写成字符串，禁止遗漏字段或多写括号。
2. swipe 必须同时包含 x1、y1、x2、y2、duration_ms 五个数字字段。不要输出 SCROLL、swipe_up 或 direction 等其他格式。
3. 不要输出 CLICK、TYPE [...]、PRESS_BACK、PRESS_HOME、COMPLETE、open、openApp 等纯文本命令或其他 JSON action 名；它们不是本项目的 action 格式。
4. 每次只输出一个动作，再根据下一张截图继续判断；不要在一个 JSON 中组合多个动作。
5. 如果无法确定精确坐标，也必须给出最可能控件中心的整数坐标，不要用文字解释替代 JSON。

手机操作规则：
1. 要查看页面下方内容时，手指从屏幕下方向上滑，例如从 (500,700) 滑到 (500,300)；要返回页面上方则反向滑动。横向列表使用水平滑动。
2. 输入文字前先确认输入框已获得焦点。软键盘会遮挡屏幕下半部分；必要时用 back 收起键盘或关闭当前弹层。
3. 遇到权限提示、隐私提示、更新提示、广告或登录弹窗时，先处理当前可见弹窗，再继续任务；不要重复已经无效的历史动作。
4. 只有当前截图已能确认任务目标达成时才输出 complete；确实无法继续时才输出 impossible；界面仍在加载时使用 wait。
"""


def build_phone_prompt(
    task: str,
    history: Optional[Any] = None,
    steplist: Optional[Any] = None,
) -> str:
    """Add task-local context to the shared phone action contract."""
    sections = [PHONE_SYSTEM_PROMPT, f"当前任务：{task}"]
    if history is not None:
        sections.append(f"历史动作：{json.dumps(history, ensure_ascii=False)}")
    if steplist:
        sections.append(f"参考步骤：{steplist}")
    return "\n\n".join(sections)
