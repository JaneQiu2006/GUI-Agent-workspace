"""HuggingFace/Transformers baseline for static GUI screenshots.

The code mirrors qwen_agent.py for prompt/message construction and reuses the
JSON action parser from benchmark_runner.py.  It intentionally keeps model
loading, preprocessing, generation, and postprocessing as separate functions so
profiling or acceleration code can be inserted later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmark_runner import parse_action
from cache_inference import PageLevelCache
from phone_prompt import build_phone_prompt


DEFAULT_MODEL_PATH = "/data2/home/models/Qwen3.8-27B"
VISION_TOKEN_MODES: Dict[str, Dict[str, int]] = {
    "default": {},
    "mild_reduce": {"max_pixels": 768 * 28 * 28},
    "aggressive_reduce": {"max_pixels": 512 * 28 * 28},
    "dynamic_safe": {},
    "dynamic_aggressive": {},
}
STATIC_VISION_TOKEN_MODES = {"default", "mild_reduce", "aggressive_reduce"}
GENERATION_PROFILE_MODES = ("generate", "manual_greedy")
InferItem = Tuple[Path, str, Optional[List[Dict[str, Any]]], Optional[Any], Optional[str]]


@dataclass
class GuiInferenceResult:
    raw_response: str
    parsed_action: Dict[str, Any]
    latency_seconds: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    prompt: str
    cache: Optional[Dict[str, Any]] = field(default=None, kw_only=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GuiProfiledInferenceResult(GuiInferenceResult):
    timings: Dict[str, float]
    memory: Dict[str, Any]
    generation_profile: Optional[Dict[str, Any]] = None


def build_gui_messages(
    image_path: Path,
    instruction: str,
    history: Optional[List[Dict[str, Any]]] = None,
    low_level: Optional[Any] = None,
    visual_token_mode: str = "default",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    action_hint: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Build the same user message shape used by qwen_agent.QwenAgent."""
    prompt = build_phone_prompt(instruction, history, low_level)
    image_content: Dict[str, Any] = {"type": "image", "image": str(image_path)}
    image_content.update(
        _visual_token_kwargs(
            visual_token_mode,
            min_pixels,
            max_pixels,
            instruction=instruction,
            action_hint=action_hint,
        )
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_content,
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
    visual_token_mode: str = "default",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    action_hint: Optional[str] = None,
) -> Tuple[Any, str, int]:
    """Apply the chat template and convert image/text into model inputs."""
    messages, prompt = build_gui_messages(
        image_path,
        instruction,
        history,
        low_level,
        visual_token_mode=visual_token_mode,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        action_hint=action_hint,
    )
    chat_text = apply_chat_template_without_thinking(processor, messages)
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


def preprocess_inputs_with_page_cache(
    processor: Any,
    image_path: Path,
    instruction: str,
    page_cache: Optional[PageLevelCache],
    cache_trajectory_id: Optional[Any] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    low_level: Optional[Any] = None,
    visual_token_mode: str = "default",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    action_hint: Optional[str] = None,
) -> Tuple[Any, str, int, Optional[Dict[str, Any]]]:
    """Apply preprocessing with optional exact page-level processor input cache."""
    if page_cache is None or not page_cache.enabled:
        inputs, prompt, input_tokens = preprocess_inputs(
            processor,
            image_path,
            instruction,
            history=history,
            low_level=low_level,
            visual_token_mode=visual_token_mode,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            action_hint=action_hint,
        )
        return inputs, prompt, input_tokens, None

    messages, prompt = build_gui_messages(
        image_path,
        instruction,
        history,
        low_level,
        visual_token_mode=visual_token_mode,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        action_hint=action_hint,
    )
    chat_text = apply_chat_template_without_thinking(processor, messages)
    fingerprint, probe = page_cache.begin_step(image_path, cache_trajectory_id)
    resolved_mode = resolve_visual_token_mode(
        visual_token_mode,
        instruction=instruction,
        action_hint=action_hint,
    )
    cache_key = page_cache.processor_key(
        chat_text,
        fingerprint,
        visual_token_mode=resolved_mode,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    inputs = page_cache.get_processor_inputs(cache_key)
    if inputs is not None:
        probe.processor_cache_hit = True
        input_tokens = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else 0
        page_cache.finish_step(fingerprint, cache_trajectory_id, probe)
        return inputs, prompt, input_tokens, probe.to_dict()

    image_inputs, video_inputs = _process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_tokens = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else 0
    page_cache.put_processor_inputs(cache_key, inputs)
    page_cache.finish_step(fingerprint, cache_trajectory_id, probe)
    return inputs, prompt, input_tokens, probe.to_dict()


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
            **_generation_token_kwargs(processor),
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


def apply_chat_template_without_thinking(processor: Any, messages: List[Dict[str, Any]]) -> str:
    """Use Qwen-style no-thinking templates when the processor supports it."""
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def infer_one(
    model: Any,
    processor: Any,
    image_path: Path,
    instruction: str,
    max_new_tokens: int = 128,
    device: str = "auto",
    history: Optional[List[Dict[str, Any]]] = None,
    low_level: Optional[Any] = None,
    visual_token_mode: str = "default",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    action_hint: Optional[str] = None,
    page_cache: Optional[PageLevelCache] = None,
    cache_trajectory_id: Optional[Any] = None,
) -> GuiInferenceResult:
    inputs, prompt, input_tokens, cache_record = preprocess_inputs_with_page_cache(
        processor,
        image_path,
        instruction,
        page_cache=page_cache,
        cache_trajectory_id=cache_trajectory_id,
        history=history,
        low_level=low_level,
        visual_token_mode=visual_token_mode,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        action_hint=action_hint,
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
        cache=cache_record,
    )


def infer_batch(
    model: Any,
    processor: Any,
    items: Sequence[Tuple[Any, ...]],
    max_new_tokens: int = 128,
    device: str = "auto",
    visual_token_mode: str = "default",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> List[GuiInferenceResult]:
    if not items:
        return []
    inputs, prompts, input_tokens = preprocess_batch_inputs(
        processor,
        items,
        visual_token_mode=visual_token_mode,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    raw_responses, latency, output_tokens = generate_batch_response(
        model,
        processor,
        inputs,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    per_item_latency = latency / len(items)
    return [
        GuiInferenceResult(
            raw_response=raw_response,
            parsed_action=postprocess_response(raw_response),
            latency_seconds=per_item_latency,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            prompt=prompt,
        )
        for raw_response, prompt, input_token_count, output_token_count in zip(
            raw_responses,
            prompts,
            input_tokens,
            output_tokens,
        )
    ]


def preprocess_batch_inputs(
    processor: Any,
    items: Sequence[Tuple[Any, ...]],
    visual_token_mode: str = "default",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[Any, List[str], List[int]]:
    prompts = []
    batch_messages = []
    chat_texts = []
    for item in items:
        image_path, instruction, history, low_level, action_hint = _unpack_infer_item(item)
        messages, prompt = build_gui_messages(
            image_path,
            instruction,
            history=history,
            low_level=low_level,
            visual_token_mode=visual_token_mode,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            action_hint=action_hint,
        )
        prompts.append(prompt)
        batch_messages.append(messages)
        chat_texts.append(apply_chat_template_without_thinking(processor, messages))

    image_inputs, video_inputs = _process_vision_info_for_batch(batch_messages)
    old_padding_side = _set_processor_padding_side(processor, "left")
    try:
        inputs = processor(
            text=chat_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    finally:
        _restore_processor_padding_side(processor, old_padding_side)
    input_tokens = []
    if hasattr(inputs, "attention_mask"):
        input_tokens = [int(value) for value in inputs.attention_mask.sum(dim=1).tolist()]
    elif hasattr(inputs, "input_ids"):
        input_tokens = [int(inputs.input_ids.shape[-1])] * len(items)
    else:
        input_tokens = [0] * len(items)
    return inputs, prompts, input_tokens


def generate_batch_response(
    model: Any,
    processor: Any,
    inputs: Any,
    max_new_tokens: int = 128,
    device: str = "auto",
) -> Tuple[List[str], float, List[int]]:
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
            **_generation_token_kwargs(processor),
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
    output_tokens = [int(ids.shape[-1]) for ids in generated_ids_trimmed]
    return [text.strip() for text in output_text], latency, output_tokens


def profile_infer_one(
    model: Any,
    processor: Any,
    image_path: Path,
    instruction: str,
    max_new_tokens: int = 128,
    device: str = "auto",
    history: Optional[List[Dict[str, Any]]] = None,
    low_level: Optional[Any] = None,
    visual_token_mode: str = "default",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    action_hint: Optional[str] = None,
    page_cache: Optional[PageLevelCache] = None,
    cache_trajectory_id: Optional[Any] = None,
    generation_profile_mode: str = "generate",
) -> GuiProfiledInferenceResult:
    import torch

    timings: Dict[str, float] = {}
    total_started = time.perf_counter()

    stage_started = time.perf_counter()
    messages, prompt = build_gui_messages(
        image_path,
        instruction,
        history,
        low_level,
        visual_token_mode=visual_token_mode,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        action_hint=action_hint,
    )
    timings["build_prompt_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    chat_text = apply_chat_template_without_thinking(processor, messages)
    timings["apply_chat_template_seconds"] = time.perf_counter() - stage_started

    cache_record: Optional[Dict[str, Any]] = None
    if page_cache is not None and page_cache.enabled:
        fingerprint, probe = page_cache.begin_step(image_path, cache_trajectory_id)
        resolved_mode = resolve_visual_token_mode(
            visual_token_mode,
            instruction=instruction,
            action_hint=action_hint,
        )
        cache_key = page_cache.processor_key(
            chat_text,
            fingerprint,
            visual_token_mode=resolved_mode,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        inputs = page_cache.get_processor_inputs(cache_key)
        if inputs is not None:
            probe.processor_cache_hit = True
            timings["vision_preprocess_seconds"] = 0.0
            timings["processor_encode_seconds"] = 0.0
        else:
            stage_started = time.perf_counter()
            image_inputs, video_inputs = _process_vision_info(messages)
            timings["vision_preprocess_seconds"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            inputs = processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            timings["processor_encode_seconds"] = time.perf_counter() - stage_started
            page_cache.put_processor_inputs(cache_key, inputs)
        page_cache.finish_step(fingerprint, cache_trajectory_id, probe)
        cache_record = probe.to_dict()
        timings["cache_lookup_seconds"] = float(cache_record.get("cache_lookup_seconds", 0.0))
        timings["cache_write_seconds"] = float(cache_record.get("cache_write_seconds", 0.0))
    else:
        stage_started = time.perf_counter()
        image_inputs, video_inputs = _process_vision_info(messages)
        timings["vision_preprocess_seconds"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        timings["processor_encode_seconds"] = time.perf_counter() - stage_started
    input_tokens = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else 0

    target_device = _input_device(model, device)
    stage_started = time.perf_counter()
    if hasattr(inputs, "to") and target_device is not None:
        inputs = inputs.to(target_device)
    _sync_if_cuda(torch, target_device)
    timings["input_to_device_seconds"] = time.perf_counter() - stage_started

    if generation_profile_mode not in GENERATION_PROFILE_MODES:
        raise ValueError(f"Unsupported generation_profile_mode: {generation_profile_mode}")

    if generation_profile_mode == "manual_greedy":
        generated_ids, generation_profile = manual_greedy_generate(
            model,
            processor,
            inputs,
            max_new_tokens=max_new_tokens,
            torch=torch,
            target_device=target_device,
        )
        timings.update(_manual_generation_timings(generation_profile))
    else:
        stage_started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **_generation_token_kwargs(processor),
            )
        _sync_if_cuda(torch, target_device)
        timings["generate_seconds"] = time.perf_counter() - stage_started
        generation_profile = {"mode": "generate"}

    stage_started = time.perf_counter()
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    raw_response = output_text[0].strip() if output_text else ""
    output_tokens = int(generated_ids_trimmed[0].shape[-1]) if generated_ids_trimmed else 0
    timings["decode_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    parsed_action = postprocess_response(raw_response)
    timings["postprocess_seconds"] = time.perf_counter() - stage_started
    timings["total_seconds"] = time.perf_counter() - total_started

    return GuiProfiledInferenceResult(
        raw_response=raw_response,
        parsed_action=parsed_action,
        latency_seconds=timings["generate_seconds"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prompt=prompt,
        cache=cache_record,
        timings=timings,
        memory=gpu_memory_snapshot(),
        generation_profile=generation_profile,
    )


def profile_infer_batch(
    model: Any,
    processor: Any,
    items: Sequence[Tuple[Any, ...]],
    max_new_tokens: int = 128,
    device: str = "auto",
    visual_token_mode: str = "default",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> List[GuiProfiledInferenceResult]:
    import torch

    if not items:
        return []

    timings: Dict[str, float] = {}
    total_started = time.perf_counter()

    stage_started = time.perf_counter()
    prompts = []
    batch_messages = []
    for item in items:
        image_path, instruction, history, low_level, action_hint = _unpack_infer_item(item)
        messages, prompt = build_gui_messages(
            image_path,
            instruction,
            history,
            low_level,
            visual_token_mode=visual_token_mode,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            action_hint=action_hint,
        )
        prompts.append(prompt)
        batch_messages.append(messages)
    timings["build_prompt_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    chat_texts = [
        apply_chat_template_without_thinking(processor, messages)
        for messages in batch_messages
    ]
    timings["apply_chat_template_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    image_inputs, video_inputs = _process_vision_info_for_batch(batch_messages)
    timings["vision_preprocess_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    old_padding_side = _set_processor_padding_side(processor, "left")
    try:
        inputs = processor(
            text=chat_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    finally:
        _restore_processor_padding_side(processor, old_padding_side)
    timings["processor_encode_seconds"] = time.perf_counter() - stage_started
    if hasattr(inputs, "attention_mask"):
        input_tokens = [int(value) for value in inputs.attention_mask.sum(dim=1).tolist()]
    elif hasattr(inputs, "input_ids"):
        input_tokens = [int(inputs.input_ids.shape[-1])] * len(items)
    else:
        input_tokens = [0] * len(items)

    target_device = _input_device(model, device)
    stage_started = time.perf_counter()
    if hasattr(inputs, "to") and target_device is not None:
        inputs = inputs.to(target_device)
    _sync_if_cuda(torch, target_device)
    timings["input_to_device_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **_generation_token_kwargs(processor),
        )
    _sync_if_cuda(torch, target_device)
    timings["generate_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    raw_responses = [text.strip() for text in output_text]
    output_tokens = [int(ids.shape[-1]) for ids in generated_ids_trimmed]
    timings["decode_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    parsed_actions = [postprocess_response(raw_response) for raw_response in raw_responses]
    timings["postprocess_seconds"] = time.perf_counter() - stage_started
    timings["total_seconds"] = time.perf_counter() - total_started

    per_item_timings = {
        key: value / len(items)
        for key, value in timings.items()
    }
    per_item_latency = timings["generate_seconds"] / len(items)
    memory = gpu_memory_snapshot()
    return [
        GuiProfiledInferenceResult(
            raw_response=raw_response,
            parsed_action=parsed_action,
            latency_seconds=per_item_latency,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            prompt=prompt,
            timings=per_item_timings,
            memory=memory,
        )
        for raw_response, parsed_action, input_token_count, output_token_count, prompt in zip(
            raw_responses,
            parsed_actions,
            input_tokens,
            output_tokens,
            prompts,
        )
    ]


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


def _process_vision_info_for_batch(batch_messages: Sequence[List[Dict[str, Any]]]) -> Tuple[Any, Any]:
    try:
        from qwen_vl_utils import process_vision_info

        return process_vision_info(list(batch_messages))
    except ImportError:
        try:
            from awq.utils.qwen_vl_utils import process_vision_info

            return process_vision_info(list(batch_messages))
        except ImportError:
            return _fallback_process_vision_info_for_batch(batch_messages)
    except (TypeError, ValueError):
        return _fallback_process_vision_info_for_batch(batch_messages)


def _fallback_process_vision_info_for_batch(batch_messages: Sequence[List[Dict[str, Any]]]) -> Tuple[Any, Any]:
    image_inputs = []
    video_inputs = []
    has_video = False
    for messages in batch_messages:
        images, videos = _process_vision_info(messages)
        if images:
            image_inputs.extend(images)
        if videos:
            has_video = True
            video_inputs.extend(videos)
    return image_inputs, video_inputs if has_video else None


def _visual_token_kwargs(
    visual_token_mode: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    instruction: str = "",
    action_hint: Optional[str] = None,
) -> Dict[str, int]:
    if visual_token_mode not in VISION_TOKEN_MODES:
        raise ValueError(f"Unsupported visual_token_mode: {visual_token_mode}")
    resolved_mode = resolve_visual_token_mode(
        visual_token_mode,
        instruction=instruction,
        action_hint=action_hint,
    )
    kwargs = dict(VISION_TOKEN_MODES[resolved_mode])
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    return kwargs


def resolve_visual_token_mode(
    visual_token_mode: str,
    instruction: str = "",
    action_hint: Optional[str] = None,
) -> str:
    if visual_token_mode in STATIC_VISION_TOKEN_MODES:
        return visual_token_mode
    action = _infer_action_hint_type(action_hint, instruction)
    if visual_token_mode == "dynamic_safe":
        if action in {"CLICK", "LONG_PRESS"}:
            return "default"
        if action in {"SCROLL", "TYPE"}:
            return "mild_reduce"
        return "aggressive_reduce"
    if visual_token_mode == "dynamic_aggressive":
        if action in {"CLICK", "LONG_PRESS"}:
            return "mild_reduce"
        return "aggressive_reduce"
    raise ValueError(f"Unsupported visual_token_mode: {visual_token_mode}")


def _infer_action_hint_type(action_hint: Optional[str], instruction: str) -> str:
    text = f"{action_hint or ''}\n{instruction or ''}".upper()
    if any(keyword in text for keyword in ("LONG_PRESS", "LONG PRESS", "长按")):
        return "LONG_PRESS"
    if any(keyword in text for keyword in ("CLICK", "TAP", "点击", "点按")):
        return "CLICK"
    if any(keyword in text for keyword in ("SCROLL", "SWIPE", "滑动", "滚动", "上滑", "下滑", "左滑", "右滑")):
        return "SCROLL"
    if any(keyword in text for keyword in ("TYPE", "INPUT_TEXT", "输入")):
        return "TYPE"
    if any(keyword in text for keyword in ("PRESS_BACK", "NAVIGATE_BACK", "返回")):
        return "PRESS_BACK"
    if any(keyword in text for keyword in ("PRESS_HOME", "NAVIGATE_HOME", "主页", "HOME")):
        return "PRESS_HOME"
    if any(keyword in text for keyword in ("OPEN_APP", "LAUNCH", "打开应用")):
        return "OPEN_APP"
    if any(keyword in text for keyword in ("WAIT", "等待")):
        return "WAIT"
    return "UNKNOWN"


def _unpack_infer_item(item: Tuple[Any, ...]) -> InferItem:
    if len(item) == 4:
        image_path, instruction, history, low_level = item
        return Path(image_path), str(instruction), history, low_level, None
    if len(item) == 5:
        image_path, instruction, history, low_level, action_hint = item
        return Path(image_path), str(instruction), history, low_level, None if action_hint is None else str(action_hint)
    raise ValueError(f"Expected infer item with 4 or 5 fields, got {len(item)}")


def _processor_tokenizer(processor: Any) -> Optional[Any]:
    return getattr(processor, "tokenizer", processor)


def manual_greedy_generate(
    model: Any,
    processor: Any,
    inputs: Any,
    max_new_tokens: int,
    torch: Any,
    target_device: Optional[Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Profiling-only greedy decode that exposes prefill and per-token timing."""
    input_ids = getattr(inputs, "input_ids", None)
    if input_ids is None:
        raise ValueError("manual_greedy generation requires inputs.input_ids")
    if int(input_ids.shape[0]) != 1:
        raise ValueError("manual_greedy generation currently supports batch_size=1 only")
    if max_new_tokens <= 0:
        return input_ids, {
            "mode": "manual_greedy",
            "prefill_seconds": 0.0,
            "ttft_seconds": 0.0,
            "decode_step_seconds": [],
            "decode_step_count": 0,
            "decode_loop_seconds": 0.0,
            "generate_seconds": 0.0,
            "stop_reason": "max_new_tokens",
        }

    input_dict = _as_input_dict(inputs)
    attention_mask = input_dict.get("attention_mask")
    generated_tokens = []
    decode_step_seconds: List[float] = []
    eos_token_ids = _eos_token_ids(processor)
    stop_reason = "max_new_tokens"

    _sync_if_cuda(torch, target_device)
    generate_started = time.perf_counter()
    prefill_started = generate_started
    with torch.inference_mode():
        outputs = _model_forward(
            model,
            {
                **input_dict,
                "use_cache": True,
                "return_dict": True,
            },
        )
        _sync_if_cuda(torch, target_device)
        prefill_seconds = time.perf_counter() - prefill_started
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _sync_if_cuda(torch, target_device)
        ttft_seconds = time.perf_counter() - generate_started
        generated_tokens.append(next_token)

        if _is_eos_token(next_token, eos_token_ids):
            stop_reason = "eos"
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None and max_new_tokens > 1 and stop_reason != "eos":
            raise RuntimeError("manual_greedy generation requires model forward to return past_key_values")
        cache_position = _initial_cache_position(input_ids, next_token, torch)
        attention_mask = _append_attention_mask(attention_mask, next_token, torch)

        while len(generated_tokens) < max_new_tokens and stop_reason != "eos":
            step_started = time.perf_counter()
            decode_inputs = _next_token_inputs(
                model,
                input_dict,
                next_token,
                past_key_values,
                attention_mask,
                cache_position,
            )
            outputs = _model_forward(model, decode_inputs)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            _sync_if_cuda(torch, target_device)
            decode_step_seconds.append(time.perf_counter() - step_started)
            generated_tokens.append(next_token)

            if _is_eos_token(next_token, eos_token_ids):
                stop_reason = "eos"
            past_key_values = getattr(outputs, "past_key_values", past_key_values)
            cache_position = cache_position + 1 if cache_position is not None else None
            attention_mask = _append_attention_mask(attention_mask, next_token, torch)

    _sync_if_cuda(torch, target_device)
    generate_seconds = time.perf_counter() - generate_started
    generated_ids_trimmed = torch.cat(generated_tokens, dim=-1)
    generated_ids = torch.cat([input_ids, generated_ids_trimmed], dim=-1)
    return generated_ids, {
        "mode": "manual_greedy",
        "prefill_seconds": prefill_seconds,
        "ttft_seconds": ttft_seconds,
        "decode_step_seconds": decode_step_seconds,
        "decode_step_count": len(decode_step_seconds),
        "decode_loop_seconds": sum(decode_step_seconds),
        "generate_seconds": generate_seconds,
        "stop_reason": stop_reason,
    }


def _manual_generation_timings(generation_profile: Dict[str, Any]) -> Dict[str, float]:
    decode_step_seconds = [
        float(value)
        for value in generation_profile.get("decode_step_seconds", [])
    ]
    timings = {
        "generate_seconds": float(generation_profile["generate_seconds"]),
        "prefill_seconds": float(generation_profile["prefill_seconds"]),
        "ttft_seconds": float(generation_profile["ttft_seconds"]),
        "decode_loop_seconds": float(generation_profile["decode_loop_seconds"]),
        "decode_step_count": float(generation_profile["decode_step_count"]),
    }
    if decode_step_seconds:
        timings.update(
            {
                "decode_token_mean_seconds": statistics.fmean(decode_step_seconds),
                "decode_token_min_seconds": min(decode_step_seconds),
                "decode_token_max_seconds": max(decode_step_seconds),
                "decode_token_median_seconds": statistics.median(decode_step_seconds),
            }
        )
    else:
        timings.update(
            {
                "decode_token_mean_seconds": 0.0,
                "decode_token_min_seconds": 0.0,
                "decode_token_max_seconds": 0.0,
                "decode_token_median_seconds": 0.0,
            }
        )
    return timings


def _as_input_dict(inputs: Any) -> Dict[str, Any]:
    if isinstance(inputs, Mapping):
        return dict(inputs)
    if hasattr(inputs, "items"):
        return dict(inputs.items())
    return {
        key: value
        for key, value in vars(inputs).items()
        if not key.startswith("_") and value is not None
    }


def _eos_token_ids(processor: Any) -> set[int]:
    eos_token_id = _generation_token_kwargs(processor).get("eos_token_id")
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(value) for value in eos_token_id}
    return {int(eos_token_id)}


def _is_eos_token(token: Any, eos_token_ids: set[int]) -> bool:
    if not eos_token_ids:
        return False
    return int(token.reshape(-1)[0].item()) in eos_token_ids


def _initial_cache_position(input_ids: Any, next_token: Any, torch: Any) -> Any:
    return torch.arange(
        int(input_ids.shape[-1]),
        int(input_ids.shape[-1]) + int(next_token.shape[-1]),
        device=next_token.device,
    )


def _append_attention_mask(attention_mask: Any, next_token: Any, torch: Any) -> Any:
    if attention_mask is None:
        return None
    ones = torch.ones(
        (int(attention_mask.shape[0]), int(next_token.shape[-1])),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([attention_mask, ones], dim=-1)


def _next_token_inputs(
    model: Any,
    base_inputs: Dict[str, Any],
    next_token: Any,
    past_key_values: Any,
    attention_mask: Any,
    cache_position: Any,
) -> Dict[str, Any]:
    model_kwargs = {
        key: value
        for key, value in base_inputs.items()
        if key not in {"input_ids", "inputs_embeds"}
    }
    model_kwargs["past_key_values"] = past_key_values
    model_kwargs["use_cache"] = True
    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask
    if cache_position is not None:
        model_kwargs["cache_position"] = cache_position
    if hasattr(model, "prepare_inputs_for_generation"):
        try:
            prepared = model.prepare_inputs_for_generation(next_token, **model_kwargs)
            prepared["use_cache"] = True
            prepared["return_dict"] = True
            return prepared
        except TypeError:
            pass
    prepared = {
        "input_ids": next_token,
        "past_key_values": past_key_values,
        "use_cache": True,
        "return_dict": True,
    }
    if attention_mask is not None:
        prepared["attention_mask"] = attention_mask
    if cache_position is not None:
        prepared["cache_position"] = cache_position
    return prepared


def _model_forward(model: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError:
        if "cache_position" not in kwargs:
            raise
        fallback = dict(kwargs)
        fallback.pop("cache_position", None)
        return model(**fallback)


def _set_processor_padding_side(processor: Any, padding_side: str) -> Optional[str]:
    tokenizer = _processor_tokenizer(processor)
    if tokenizer is None or not hasattr(tokenizer, "padding_side"):
        return None
    old_padding_side = str(tokenizer.padding_side)
    tokenizer.padding_side = padding_side
    return old_padding_side


def _restore_processor_padding_side(processor: Any, old_padding_side: Optional[str]) -> None:
    if old_padding_side is None:
        return
    tokenizer = _processor_tokenizer(processor)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = old_padding_side


def _generation_token_kwargs(processor: Any) -> Dict[str, int]:
    tokenizer = _processor_tokenizer(processor)
    if tokenizer is None:
        return {}
    kwargs = {}
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    if pad_token_id is None and eos_token_id is not None:
        pad_token_id = eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = int(pad_token_id)
    if eos_token_id is not None:
        kwargs["eos_token_id"] = int(eos_token_id)
    return kwargs


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


def reset_gpu_memory_stats() -> List[str]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    warnings = []
    for index in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(index)
        except RuntimeError as exc:
            warnings.append(f"cuda:{index}: {exc}")
    return warnings


def gpu_memory_snapshot() -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    allocated = {}
    reserved = {}
    peak_allocated = {}
    peak_reserved = {}
    warnings = []
    for index in range(torch.cuda.device_count()):
        key = str(index)
        try:
            allocated[key] = int(torch.cuda.memory_allocated(index))
            reserved[key] = int(torch.cuda.memory_reserved(index))
            peak_allocated[key] = int(torch.cuda.max_memory_allocated(index))
            peak_reserved[key] = int(torch.cuda.max_memory_reserved(index))
        except RuntimeError as exc:
            warnings.append(f"cuda:{index}: {exc}")
    snapshot: Dict[str, Any] = {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gb": {
            key: round(value / 1024**3, 4) for key, value in peak_allocated.items()
        },
        "peak_reserved_gb": {
            key: round(value / 1024**3, 4) for key, value in peak_reserved.items()
        },
    }
    if warnings:
        snapshot["warnings"] = warnings
    return snapshot


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
