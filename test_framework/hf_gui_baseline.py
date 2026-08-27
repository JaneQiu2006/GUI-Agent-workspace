"""HuggingFace/Transformers baseline for static GUI screenshots.

The code mirrors qwen_agent.py for prompt/message construction and reuses the
JSON action parser from benchmark_runner.py.  It intentionally keeps model
loading, preprocessing, generation, and postprocessing as separate functions so
profiling or acceleration code can be inserted later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from benchmark_runner import parse_action
from phone_prompt import build_phone_prompt


DEFAULT_MODEL_PATH = "/data2/home/models/Qwen3.8-27B"


@dataclass
class GuiInferenceResult:
    raw_response: str
    parsed_action: Dict[str, Any]
    latency_seconds: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    prompt: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_gui_messages(
    image_path: Path,
    instruction: str,
    history: Optional[List[Dict[str, Any]]] = None,
    low_level: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Build the same user message shape used by qwen_agent.QwenAgent."""
    prompt = build_phone_prompt(instruction, history, low_level)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": str(image_path)},
            ],
        }
    ]
    return messages, prompt


def load_model_and_processor(
    model_path: str,
    device: str = "auto",
    device_map: Optional[str] = "auto",
    dtype: str = "auto",
    attn_implementation: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load a local Qwen-VL style model and processor without downloading."""
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_type = str(getattr(config, "model_type", "") or "")
    architectures = tuple(getattr(config, "architectures", None) or ())
    model_class_names = [
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ]
    if _looks_like_qwen25_vl(model_type, architectures):
        model_class_names.append("Qwen2_5_VLForConditionalGeneration")
    if _looks_like_qwen2_vl(model_type, architectures):
        model_class_names.append("Qwen2VLForConditionalGeneration")

    model_classes = []
    for name in model_class_names:
        candidate = getattr(transformers, name, None)
        if candidate is not None:
            model_classes.append((name, candidate))
    if not model_classes:
        raise RuntimeError(
            "当前 transformers 版本缺少可用的 VLM model class，请在服务器环境安装支持 Qwen-VL 的 transformers。"
        )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    torch_dtype = _resolve_dtype(torch, dtype)
    kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if device_map:
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    errors = []
    model = None
    for class_name, model_class in model_classes:
        try:
            model = model_class.from_pretrained(model_path, config=config, **kwargs)
            break
        except Exception as exc:
            errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
            if "attn_implementation" not in kwargs:
                continue
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("attn_implementation", None)
            try:
                model = model_class.from_pretrained(model_path, config=config, **retry_kwargs)
                break
            except Exception as retry_exc:
                errors.append(
                    f"{class_name} without attn_implementation: "
                    f"{type(retry_exc).__name__}: {retry_exc}"
                )
    if model is None:
        joined_errors = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(
            "无法加载本地多模态模型。"
            f"config.model_type={model_type!r}, architectures={architectures!r}。\n"
            "已尝试的 Transformers model class 均失败：\n"
            f"{joined_errors}"
        )

    if not device_map and device != "auto":
        model = model.to(device)
    model.eval()
    return model, processor


def preprocess_inputs(
    processor: Any,
    image_path: Path,
    instruction: str,
    history: Optional[List[Dict[str, Any]]] = None,
    low_level: Optional[Any] = None,
) -> Tuple[Any, str, int]:
    """Apply the chat template and convert image/text into model inputs."""
    messages, prompt = build_gui_messages(image_path, instruction, history, low_level)
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = _process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_tokens = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else 0
    return inputs, prompt, input_tokens


def generate_response(
    model: Any,
    processor: Any,
    inputs: Any,
    max_new_tokens: int = 128,
    device: str = "auto",
) -> Tuple[str, float, int]:
    """Run model.generate and decode only the newly generated tokens."""
    import torch

    target_device = _input_device(model, device)
    if hasattr(inputs, "to") and target_device is not None:
        inputs = inputs.to(target_device)

    _sync_if_cuda(torch, target_device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    _sync_if_cuda(torch, target_device)
    latency = time.perf_counter() - started

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output_tokens = int(generated_ids_trimmed[0].shape[-1]) if generated_ids_trimmed else 0
    return output_text[0].strip() if output_text else "", latency, output_tokens


def postprocess_response(raw_response: str) -> Dict[str, Any]:
    return parse_action(raw_response)


def infer_one(
    model: Any,
    processor: Any,
    image_path: Path,
    instruction: str,
    max_new_tokens: int = 128,
    device: str = "auto",
    history: Optional[List[Dict[str, Any]]] = None,
    low_level: Optional[Any] = None,
) -> GuiInferenceResult:
    inputs, prompt, input_tokens = preprocess_inputs(
        processor,
        image_path,
        instruction,
        history=history,
        low_level=low_level,
    )
    raw_response, latency, output_tokens = generate_response(
        model,
        processor,
        inputs,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    return GuiInferenceResult(
        raw_response=raw_response,
        parsed_action=postprocess_response(raw_response),
        latency_seconds=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prompt=prompt,
    )


def mock_infer_one(
    image_path: Path,
    instruction: str,
    raw_response: str,
    history: Optional[List[Dict[str, Any]]] = None,
    low_level: Optional[Any] = None,
) -> GuiInferenceResult:
    """Static-check path that validates prompt/action handling without loading a model."""
    _, prompt = build_gui_messages(image_path, instruction, history, low_level)
    return GuiInferenceResult(
        raw_response=raw_response,
        parsed_action=postprocess_response(raw_response),
        latency_seconds=0.0,
        input_tokens=None,
        output_tokens=None,
        prompt=prompt,
    )


def _process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[Any, Any]:
    try:
        from qwen_vl_utils import process_vision_info

        return process_vision_info(messages)
    except ImportError:
        try:
            from awq.utils.qwen_vl_utils import process_vision_info

            return process_vision_info(messages)
        except ImportError:
            from PIL import Image

            image_paths = [
                item["image"]
                for message in messages
                for item in message.get("content", [])
                if isinstance(item, dict) and item.get("type") == "image"
            ]
            return [Image.open(path).convert("RGB") for path in image_paths], None


def _resolve_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _looks_like_qwen25_vl(model_type: str, architectures: Tuple[str, ...]) -> bool:
    values = (model_type, *architectures)
    return any("qwen2_5_vl" in value.lower() or "qwen2.5vl" in value.lower() for value in values)


def _looks_like_qwen2_vl(model_type: str, architectures: Tuple[str, ...]) -> bool:
    values = (model_type, *architectures)
    return any("qwen2_vl" in value.lower() or "qwen2vl" in value.lower() for value in values)


def _input_device(model: Any, device: str) -> Optional[Any]:
    if device != "auto":
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", None)


def _sync_if_cuda(torch: Any, device: Optional[Any]) -> None:
    if device is None:
        return
    device_text = str(device)
    if device_text.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
