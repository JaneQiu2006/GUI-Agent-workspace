from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FRAMEWORK = REPO_ROOT / "test_framework"
for path in (TEST_FRAMEWORK, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cache_inference import (
    PAGE_CACHE_MODES,
    PAGE_CACHE_SCOPES,
    PAGE_CACHE_SIMILARITIES,
    PageCacheConfig,
    PageLevelCache,
    parse_normalized_bboxes,
)
from hf_gui_baseline import (
    DEFAULT_MODEL_PATH,
    GENERATION_PROFILE_MODES,
    GuiProfiledInferenceResult,
    VISION_TOKEN_MODES,
    load_model_and_processor,
    mock_infer_one,
    profile_infer_one,
)
from phone_prompt import build_phone_prompt


DEFAULT_CONFIG = REPO_ROOT / "configs" / "androidworld_cache_subset.json"
DEFAULT_TASKS = (
    "ContactsAddContact",
    "ClockTimerEntry",
    "ExpenseAddSingle",
    "MarkorCreateFolder",
    "MarkorCreateNote",
    "SimpleSmsSend",
    "SimpleSmsReply",
    "SimpleCalendarAddOneEventTomorrow",
    "SimpleCalendarAddOneEventRelativeDay",
    "SimpleCalendarDeleteOneEvent",
)
TASK_APP_HINTS = {
    "Contacts": "contacts",
    "Clock": "clock",
    "Expense": "expense",
    "Markor": "markor",
    "SimpleSms": "sms",
    "SimpleCalendar": "calendar",
}


@dataclass
class BenchmarkDefaults:
    tasks: Tuple[str, ...] = DEFAULT_TASKS
    n_task_combinations: int = 5
    warmup_task_random_seed: int = 30
    evaluation_task_random_seed: int = 31
    smoke_task_random_seed: int = 32
    max_steps: Optional[int] = None
    transition_pause: Optional[float] = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small cache-oriented AndroidWorld benchmark with this project's GUI inference path."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--android_world_path", type=Path, help="Path to a local android_world checkout")
    parser.add_argument("--run_mode", choices=("smoke", "baseline", "warmup", "evaluation", "warmup_eval"), default="evaluation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", help="Comma-separated AndroidWorld task templates; overrides config")
    parser.add_argument("--tasks_file", type=Path, help="One AndroidWorld task template per line")
    parser.add_argument("--n_task_combinations", type=int)
    parser.add_argument("--task_random_seed", type=int, help="Overrides the phase-specific seed")
    parser.add_argument("--warmup_task_random_seed", type=int)
    parser.add_argument("--evaluation_task_random_seed", type=int)
    parser.add_argument("--smoke_task_random_seed", type=int)
    parser.add_argument("--limit_episodes", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--transition_pause", type=float)
    parser.add_argument("--perform_emulator_setup", action="store_true")
    parser.add_argument("--adb_path", default="", help="Path to adb; empty uses AndroidWorld default")
    parser.add_argument("--console_port", type=int, default=5554)
    parser.add_argument("--grpc_port", type=int, default=8554)
    parser.add_argument("--freeze_datetime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"))
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--visual_token_mode", default="aggressive_reduce", choices=tuple(VISION_TOKEN_MODES))
    parser.add_argument("--min_pixels", type=int)
    parser.add_argument("--max_pixels", type=int)
    parser.add_argument("--generation_profile_mode", default="generate", choices=GENERATION_PROFILE_MODES)
    parser.add_argument("--mock_response", help="Use a fixed raw model response instead of loading a model")
    parser.add_argument("--page_cache_mode", default="off", choices=PAGE_CACHE_MODES)
    parser.add_argument("--page_cache_scope", default="dataset", choices=PAGE_CACHE_SCOPES)
    parser.add_argument("--page_cache_similarity", default="tile", choices=PAGE_CACHE_SIMILARITIES)
    parser.add_argument("--page_cache_max_entries", type=int, default=4096)
    parser.add_argument("--page_cache_near_dhash_threshold", type=int, default=4)
    parser.add_argument("--page_cache_near_tile_threshold", type=float, default=0.98)
    parser.add_argument("--page_cache_patch_tile_threshold", type=float, default=0.90)
    parser.add_argument("--page_cache_patch_max_changed_area_ratio", type=float, default=0.25)
    parser.add_argument("--page_cache_patch_critical_region", action="append", default=[])
    parser.add_argument("--page_cache_tile_rows", type=int, default=8)
    parser.add_argument("--page_cache_tile_cols", type=int, default=16)
    parser.add_argument("--page_cache_ignored_top_ratio", type=float, default=0.02)
    parser.add_argument("--page_cache_ignored_bottom_ratio", type=float, default=0.0)
    parser.add_argument("--cache_input", type=Path, help="Warm-up page cache records JSON/JSONL to import before evaluation")
    parser.add_argument("--cache_output", type=Path, help="Defaults to <output>/page_cache_records.jsonl")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    defaults = load_defaults(args.config)
    tasks = resolve_tasks(args, defaults)
    if not tasks:
        raise SystemExit("No AndroidWorld tasks selected")

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "run_config.json", build_run_config(args, defaults, tasks))
    page_cache = make_page_cache(args)
    imported_cache_metadata = import_page_cache(args.cache_input, page_cache) if page_cache else {}

    mock_response = args.mock_response
    if args.run_mode == "smoke" and not mock_response:
        mock_response = '{"action":"complete"}'
    model = processor = None
    if mock_response is None:
        model, processor = load_model_and_processor(
            args.model_path,
            device=args.device,
            device_map=args.device_map or None,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )

    modules = import_androidworld(args.android_world_path)
    env = load_androidworld_env(args, modules)
    agent = AndroidWorldCacheAgent(
        env=env,
        model=model,
        processor=processor,
        output_root=args.output,
        page_cache=page_cache,
        cache_metadata=imported_cache_metadata,
        mock_response=mock_response,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        visual_token_mode=args.visual_token_mode,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        generation_profile_mode=args.generation_profile_mode,
        transition_pause=resolve_transition_pause(args, defaults),
    )

    all_episodes: List[Dict[str, Any]] = []
    all_steps: List[Dict[str, Any]] = []
    try:
        for phase in phases_for_mode(args.run_mode):
            phase_seed = resolve_phase_seed(args, defaults, phase)
            n_combinations = resolve_n_task_combinations(args, defaults, phase)
            suite = create_androidworld_suite(modules, env, tasks, n_combinations, phase_seed)
            limit_episodes = args.limit_episodes
            if args.run_mode == "smoke" and limit_episodes is None:
                limit_episodes = 1
            phase_episodes, phase_steps = run_suite_phase(
                modules=modules,
                env=env,
                agent=agent,
                suite=suite,
                phase=phase,
                task_seed=phase_seed,
                max_steps_override=resolve_max_steps(args, defaults),
                limit_episodes=limit_episodes,
                output_root=args.output,
            )
            all_episodes.extend(phase_episodes)
            all_steps.extend(phase_steps)
            if args.run_mode == "smoke":
                break
    finally:
        env.close()

    cache_output = args.cache_output or (args.output / "page_cache_records.jsonl")
    if page_cache is not None:
        export_page_cache(cache_output, page_cache, agent.cache_metadata)
    summary = summarize_run(all_episodes, all_steps)
    summary["cache_store"] = page_cache.summary() if page_cache else {"mode": "off"}
    summary["cache_records_path"] = str(cache_output) if page_cache else None
    write_json(args.output / "summary.json", summary)
    print("ANDROIDWORLD_CACHE_BENCHMARK_DONE " + json.dumps(summary["metrics"], ensure_ascii=False), flush=True)
    print(f"ANDROIDWORLD_CACHE_BENCHMARK_OUTPUT {args.output / 'summary.json'}", flush=True)
    return 0


class AndroidWorldCacheAgent:
    def __init__(
        self,
        env: Any,
        model: Any,
        processor: Any,
        output_root: Path,
        page_cache: Optional[PageLevelCache],
        cache_metadata: Dict[str, Dict[str, Any]],
        mock_response: Optional[str],
        max_new_tokens: int,
        device: str,
        visual_token_mode: str,
        min_pixels: Optional[int],
        max_pixels: Optional[int],
        generation_profile_mode: str,
        transition_pause: Optional[float],
    ) -> None:
        self.env = env
        self.name = "qwen_cache_adapter"
        self.model = model
        self.processor = processor
        self.output_root = output_root
        self.page_cache = page_cache
        self.cache_metadata = dict(cache_metadata)
        self.mock_response = mock_response
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.visual_token_mode = visual_token_mode
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generation_profile_mode = generation_profile_mode
        self.transition_pause = transition_pause
        self.max_steps: Optional[int] = None
        self.history: List[Dict[str, Any]] = []
        self.episode_dir = output_root
        self.episode_meta: Dict[str, Any] = {}
        self.step_records: List[Dict[str, Any]] = []
        self._step_id = 0

    def set_episode(self, episode_dir: Path, episode_meta: Dict[str, Any]) -> None:
        self.episode_dir = episode_dir
        self.episode_meta = dict(episode_meta)
        self.step_records = []
        self._step_id = 0

    def reset(self, go_home: bool = False) -> None:
        self.env.reset(go_home=go_home)
        try:
            self.env.hide_automation_ui()
        except Exception:
            pass
        self.history = []

    def set_max_steps(self, max_steps: int) -> None:
        self.max_steps = int(max_steps)

    def step(self, goal: str) -> Any:
        from android_world.agents import base_agent

        if self.transition_pause is None:
            state = self.env.get_state(wait_to_stabilize=True)
        else:
            time.sleep(max(0.0, self.transition_pause))
            state = self.env.get_state(wait_to_stabilize=False)

        step_id = self._step_id
        self._step_id += 1
        step_started = time.perf_counter()
        screenshot_path = self.episode_dir / f"step_{step_id:03d}.png"
        save_pixels(state.pixels, screenshot_path)
        ui_elements = serialize_ui_elements(getattr(state, "ui_elements", []))
        ui_text = format_ui_elements_for_prompt(ui_elements)
        previous_action = self.history[-1]["current_action"] if self.history else None

        if self.mock_response is not None:
            result = mock_result(
                screenshot_path,
                goal,
                self.mock_response,
                self.history,
                ui_text,
                self.page_cache,
                self.episode_meta.get("episode_id"),
            )
        else:
            result = profile_infer_one(
                self.model,
                self.processor,
                screenshot_path,
                goal,
                max_new_tokens=self.max_new_tokens,
                device=self.device,
                history=self.history,
                low_level=ui_text,
                visual_token_mode=self.visual_token_mode,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                page_cache=self.page_cache,
                cache_trajectory_id=self.episode_meta.get("episode_id"),
                generation_profile_mode=self.generation_profile_mode,
            )

        current_action = dict(result.parsed_action)
        converted_action, done = convert_to_androidworld_action(current_action, self.env.logical_screen_size)
        execution_error = None
        if not done:
            try:
                self.env.execute_action(converted_action)
            except Exception as exc:
                execution_error = f"{type(exc).__name__}: {exc}"
                done = True

        cache_record = dict(result.cache) if isinstance(result.cache, dict) else None
        cache_group = cache_reuse_group(cache_record, self.episode_meta, self.cache_metadata)
        if cache_record and cache_record.get("image_sha256"):
            self.cache_metadata[str(cache_record["image_sha256"])] = {
                "phase": self.episode_meta.get("phase"),
                "task_template": self.episode_meta.get("task_template"),
                "app": self.episode_meta.get("app"),
                "episode_id": self.episode_meta.get("episode_id"),
                "task_seed": self.episode_meta.get("task_seed"),
                "goal": goal,
            }
        step_record = {
            **self.episode_meta,
            "step_id": step_id,
            "timestamp": utc_now(),
            "task_goal": goal,
            "screenshot": str(screenshot_path),
            "ui_hierarchy": {
                "ui_elements": ui_elements,
                "forest_available": getattr(state, "forest", None) is not None,
            },
            "previous_action": previous_action,
            "current_action": current_action,
            "androidworld_action": converted_action.as_dict(skip_none=True),
            "model_input": result.prompt,
            "model_output": result.raw_response,
            "parsed_action": current_action,
            "inference_latency_seconds": result.latency_seconds,
            "step_latency_seconds": time.perf_counter() - step_started,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "timings": getattr(result, "timings", {}),
            "cache": cache_record,
            "cache_hit": bool(cache_record and (cache_record.get("page_cache_hit") or cache_record.get("processor_cache_hit"))),
            "cache_lookup_latency_seconds": float(cache_record.get("cache_lookup_seconds", 0.0)) if cache_record else 0.0,
            "cache_hit_entry": cache_hit_entry(cache_record, self.cache_metadata),
            "cache_reuse_group": cache_group,
            "fallback_to_inference": True,
            "model_invoked": self.mock_response is None,
            "execution_error": execution_error,
        }
        self.history.append({
            "current_action": current_action,
            "androidworld_action": converted_action.as_dict(skip_none=True),
            "execution_error": execution_error,
        })
        self.step_records.append(step_record)
        return base_agent.AgentInteractionResult(done=done, data=step_record)


def load_defaults(path: Path) -> BenchmarkDefaults:
    if not path.is_file():
        return BenchmarkDefaults()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return BenchmarkDefaults(
        tasks=tuple(str(value) for value in data.get("tasks", DEFAULT_TASKS)),
        n_task_combinations=int(data.get("n_task_combinations", 5)),
        warmup_task_random_seed=int(data.get("warmup_task_random_seed", 30)),
        evaluation_task_random_seed=int(data.get("evaluation_task_random_seed", 31)),
        smoke_task_random_seed=int(data.get("smoke_task_random_seed", 32)),
        max_steps=data.get("max_steps"),
        transition_pause=data.get("transition_pause"),
    )


def resolve_tasks(args: argparse.Namespace, defaults: BenchmarkDefaults) -> List[str]:
    if args.tasks:
        return [part.strip() for part in args.tasks.split(",") if part.strip()]
    if args.tasks_file:
        return [
            line.strip()
            for line in args.tasks_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return list(defaults.tasks)


def phases_for_mode(run_mode: str) -> Tuple[str, ...]:
    if run_mode == "baseline":
        return ("evaluation",)
    if run_mode == "warmup_eval":
        return ("warmup", "evaluation")
    if run_mode == "smoke":
        return ("smoke",)
    return (run_mode,)


def resolve_phase_seed(args: argparse.Namespace, defaults: BenchmarkDefaults, phase: str) -> int:
    if args.task_random_seed is not None:
        return args.task_random_seed
    if phase == "warmup":
        return args.warmup_task_random_seed if args.warmup_task_random_seed is not None else defaults.warmup_task_random_seed
    if phase == "smoke":
        return args.smoke_task_random_seed if args.smoke_task_random_seed is not None else defaults.smoke_task_random_seed
    return args.evaluation_task_random_seed if args.evaluation_task_random_seed is not None else defaults.evaluation_task_random_seed


def resolve_n_task_combinations(args: argparse.Namespace, defaults: BenchmarkDefaults, phase: str) -> int:
    if phase == "smoke":
        return 1 if args.n_task_combinations is None else max(1, args.n_task_combinations)
    return max(1, args.n_task_combinations if args.n_task_combinations is not None else defaults.n_task_combinations)


def resolve_max_steps(args: argparse.Namespace, defaults: BenchmarkDefaults) -> Optional[int]:
    return args.max_steps if args.max_steps is not None else defaults.max_steps


def resolve_transition_pause(args: argparse.Namespace, defaults: BenchmarkDefaults) -> Optional[float]:
    return args.transition_pause if args.transition_pause is not None else defaults.transition_pause


def import_androidworld(android_world_path: Optional[Path]) -> Dict[str, Any]:
    if android_world_path:
        sys.path.insert(0, str(android_world_path.resolve()))
    from android_world import registry
    from android_world import suite_utils
    from android_world.env import env_launcher

    return {"registry": registry, "suite_utils": suite_utils, "env_launcher": env_launcher}


def load_androidworld_env(args: argparse.Namespace, modules: Dict[str, Any]) -> Any:
    kwargs = {
        "console_port": args.console_port,
        "emulator_setup": args.perform_emulator_setup,
        "freeze_datetime": args.freeze_datetime,
        "grpc_port": args.grpc_port,
    }
    if args.adb_path:
        kwargs["adb_path"] = args.adb_path
    return modules["env_launcher"].load_and_setup_env(**kwargs)


def create_androidworld_suite(
    modules: Dict[str, Any],
    env: Any,
    tasks: Sequence[str],
    n_task_combinations: int,
    task_seed: int,
) -> Any:
    registry_module = modules["registry"]
    task_registry = registry_module.TaskRegistry()
    android_family = task_registry.ANDROID_WORLD_FAMILY
    aw_registry = task_registry.get_registry(family=android_family)
    suite = modules["suite_utils"].create_suite(
        aw_registry,
        n_task_combinations=n_task_combinations,
        seed=task_seed,
        tasks=list(tasks),
        env=env,
    )
    suite.suite_family = android_family
    return suite


def run_suite_phase(
    modules: Dict[str, Any],
    env: Any,
    agent: AndroidWorldCacheAgent,
    suite: Any,
    phase: str,
    task_seed: int,
    max_steps_override: Optional[int],
    limit_episodes: Optional[int],
    output_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    episodes = []
    steps = []
    episode_index = 0
    for task_template, instances in suite.items():
        for instance_index, task in enumerate(instances):
            if limit_episodes is not None and episode_index >= limit_episodes:
                return episodes, steps
            episode_id = f"{phase}_{task_template}_{instance_index:03d}"
            episode_dir = output_root / "episodes" / episode_id
            episode_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "phase": phase,
                "episode_id": episode_id,
                "episode_index": episode_index,
                "task_template": task_template,
                "task_instance_index": instance_index,
                "task_seed": task_seed,
                "task_params": sanitize(getattr(task, "params", {})),
                "task_goal": str(getattr(task, "goal", "")),
                "app": app_for_task(task_template),
            }
            agent.set_episode(episode_dir, meta)
            started = time.perf_counter()
            exception_info = None
            done = False
            reward = 0.0
            try:
                task.initialize_task(env)
                start_on_home = bool(getattr(task, "start_on_home_screen", False))
                agent.reset(go_home=start_on_home)
                max_steps = max_steps_override or int(float(getattr(task, "complexity", 1.0)) * 10)
                agent.set_max_steps(max_steps)
                for _ in range(max_steps):
                    result = agent.step(str(task.goal))
                    if result.done:
                        done = True
                        break
                reward = float(task.is_successful(env)) if done else 0.0
            except Exception as exc:
                exception_info = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    task.tear_down(env)
                except Exception:
                    pass
            success = bool(done and reward > 0.5 and exception_info is None)
            episode_seconds = time.perf_counter() - started
            episode_summary = {
                **meta,
                "done": done,
                "task_final_reward": reward,
                "success": success,
                "episode_latency_seconds": episode_seconds,
                "episode_length": len(agent.step_records),
                "exception_info": exception_info,
                "finished_at": utc_now(),
            }
            for step_record in agent.step_records:
                step_record["task_final_reward"] = reward
                step_record["success"] = success
                step_record["episode_latency_seconds"] = episode_seconds
                append_jsonl(output_root / "steps.jsonl", step_record)
                steps.append(step_record)
            append_jsonl(output_root / "episodes.jsonl", episode_summary)
            write_json(episode_dir / "episode_summary.json", episode_summary)
            episodes.append(episode_summary)
            episode_index += 1
            print(
                f"ANDROIDWORLD_EPISODE_DONE phase={phase} episode={episode_id} "
                f"success={success} reward={reward} steps={len(agent.step_records)}",
                flush=True,
            )
    return episodes, steps


def make_page_cache(args: argparse.Namespace) -> Optional[PageLevelCache]:
    if args.page_cache_mode == "off":
        return None
    identity = "|".join(
        [
            str(args.model_path),
            str(args.device),
            str(args.device_map),
            str(args.dtype),
            str(args.attn_implementation),
            str(args.visual_token_mode),
            str(args.min_pixels),
            str(args.max_pixels),
        ]
    )
    return PageLevelCache(
        PageCacheConfig(
            mode=args.page_cache_mode,
            scope=args.page_cache_scope,
            similarity=args.page_cache_similarity,
            max_entries=args.page_cache_max_entries,
            near_dhash_threshold=args.page_cache_near_dhash_threshold,
            near_tile_threshold=args.page_cache_near_tile_threshold,
            patch_tile_threshold=args.page_cache_patch_tile_threshold,
            patch_max_changed_area_ratio=args.page_cache_patch_max_changed_area_ratio,
            patch_critical_regions=parse_normalized_bboxes(args.page_cache_patch_critical_region),
            tile_rows=args.page_cache_tile_rows,
            tile_cols=args.page_cache_tile_cols,
            ignored_top_ratio=args.page_cache_ignored_top_ratio,
            ignored_bottom_ratio=args.page_cache_ignored_bottom_ratio,
            identity=identity,
        )
    )


def import_page_cache(path: Optional[Path], page_cache: Optional[PageLevelCache]) -> Dict[str, Dict[str, Any]]:
    if path is None or page_cache is None or not path.is_file():
        return {}
    records = read_cache_records(path)
    page_cache.import_page_records(records)
    metadata = {}
    for record in records:
        fingerprint = record.get("fingerprint", record)
        item_metadata = record.get("metadata")
        if isinstance(fingerprint, dict) and isinstance(item_metadata, dict):
            metadata[str(fingerprint.get("image_sha256"))] = dict(item_metadata)
    print(f"ANDROIDWORLD_CACHE_IMPORTED records={len(records)} path={path}", flush=True)
    return metadata


def export_page_cache(path: Path, page_cache: PageLevelCache, metadata: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in page_cache.export_page_records():
            fingerprint = record.get("fingerprint", {})
            image_sha = str(fingerprint.get("image_sha256", ""))
            enriched = dict(record)
            if image_sha in metadata:
                enriched["metadata"] = metadata[image_sha]
            file.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def read_cache_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("{") or text.startswith("["):
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("records", [])
        return [record for record in data if isinstance(record, dict)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def mock_result(
    image_path: Path,
    goal: str,
    raw_response: str,
    history: List[Dict[str, Any]],
    ui_text: str,
    page_cache: Optional[PageLevelCache],
    trajectory_id: Optional[Any],
) -> GuiProfiledInferenceResult:
    result = mock_infer_one(image_path, goal, raw_response, history=history, low_level=ui_text)
    cache_record = None
    if page_cache is not None and page_cache.enabled:
        fingerprint, probe = page_cache.begin_step(image_path, trajectory_id)
        page_cache.finish_step(fingerprint, trajectory_id, probe)
        cache_record = probe.to_dict()
    return GuiProfiledInferenceResult(
        raw_response=result.raw_response,
        parsed_action=result.parsed_action,
        latency_seconds=0.0,
        input_tokens=None,
        output_tokens=None,
        prompt=build_phone_prompt(goal, history, ui_text),
        cache=cache_record,
        timings={
            "cache_lookup_seconds": float(cache_record.get("cache_lookup_seconds", 0.0)) if cache_record else 0.0,
            "cache_write_seconds": float(cache_record.get("cache_write_seconds", 0.0)) if cache_record else 0.0,
            "generate_seconds": 0.0,
            "total_seconds": 0.0,
        },
        memory={},
        generation_profile={"mode": "mock"},
    )


def convert_to_androidworld_action(action: Dict[str, Any], screen_size: Tuple[int, int]) -> Tuple[Any, bool]:
    from android_world.env import json_action

    kind = str(action.get("action", "")).strip().lower()
    if kind == "tap":
        return json_action.JSONAction(
            action_type=json_action.CLICK,
            x=screen_coordinate(action.get("x"), screen_size[0]),
            y=screen_coordinate(action.get("y"), screen_size[1]),
        ), False
    if kind in {"swipe", "scroll"} or kind.startswith("swipe_") or kind.startswith("scroll_"):
        return json_action.JSONAction(action_type=json_action.SWIPE, direction=swipe_direction(action)), False
    if kind == "type":
        return json_action.JSONAction(action_type=json_action.INPUT_TEXT, text=str(action.get("text", ""))), False
    if kind == "back":
        return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK), False
    if kind == "home":
        return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME), False
    if kind == "wait":
        return json_action.JSONAction(action_type=json_action.WAIT), False
    if kind == "complete":
        return json_action.JSONAction(action_type=json_action.STATUS, goal_status="complete"), True
    if kind == "impossible":
        return json_action.JSONAction(action_type=json_action.STATUS, goal_status="infeasible"), True
    return json_action.JSONAction(action_type=json_action.UNKNOWN), False


def screen_coordinate(value: Any, dimension: int) -> int:
    number = float(value)
    normalized = number if 0 <= number <= 1 else number / 1000.0
    return int(max(0.0, min(1.0, normalized)) * dimension)


def swipe_direction(action: Dict[str, Any]) -> str:
    direction = str(action.get("direction") or "").lower()
    if direction in {"up", "down", "left", "right"}:
        return direction
    try:
        x1 = float(action.get("x1", 500))
        y1 = float(action.get("y1", 700))
        x2 = float(action.get("x2", 500))
        y2 = float(action.get("y2", 300))
    except (TypeError, ValueError):
        return "up"
    dx = x2 - x1
    dy = y2 - y1
    if abs(dy) >= abs(dx):
        return "up" if dy < 0 else "down"
    return "left" if dx < 0 else "right"


def save_pixels(pixels: Any, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).convert("RGB").save(path, format="PNG")


def serialize_ui_elements(ui_elements: Iterable[Any]) -> List[Dict[str, Any]]:
    records = []
    for index, element in enumerate(ui_elements):
        record = {"index": index}
        for key in (
            "text",
            "content_description",
            "class_name",
            "hint_text",
            "is_checked",
            "is_checkable",
            "is_clickable",
            "is_editable",
            "is_enabled",
            "is_focused",
            "is_focusable",
            "is_long_clickable",
            "is_scrollable",
            "is_selected",
            "is_visible",
            "package_name",
            "resource_name",
            "resource_id",
        ):
            value = getattr(element, key, None)
            if value is not None:
                record[key] = sanitize(value)
        bbox = getattr(element, "bbox_pixels", None) or getattr(element, "bbox", None)
        if bbox is not None:
            record["bbox"] = {
                "x_min": getattr(bbox, "x_min", None),
                "x_max": getattr(bbox, "x_max", None),
                "y_min": getattr(bbox, "y_min", None),
                "y_max": getattr(bbox, "y_max", None),
            }
        records.append(record)
    return records


def format_ui_elements_for_prompt(ui_elements: List[Dict[str, Any]], limit: int = 80) -> str:
    lines = ["当前 Android accessibility/UI hierarchy 摘要："]
    for element in ui_elements[:limit]:
        compact = {
            key: element.get(key)
            for key in ("index", "text", "content_description", "class_name", "bbox", "is_clickable", "is_editable", "is_scrollable")
            if element.get(key) not in (None, "", False)
        }
        lines.append(json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
    if len(ui_elements) > limit:
        lines.append(f"... {len(ui_elements) - limit} more elements")
    return "\n".join(lines)


def cache_hit_entry(cache_record: Optional[Dict[str, Any]], metadata: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cache_record:
        return None
    image_sha = cache_record.get("matched_image_sha256")
    if not image_sha:
        patch_diff = cache_record.get("patch_diff") or {}
        image_sha = patch_diff.get("base_image_sha256")
    if not image_sha:
        return None
    return {"image_sha256": image_sha, "metadata": metadata.get(str(image_sha), {})}


def cache_reuse_group(
    cache_record: Optional[Dict[str, Any]],
    current_meta: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
) -> str:
    if not cache_record or not cache_record.get("page_cache_hit"):
        return "miss"
    hit = cache_hit_entry(cache_record, metadata)
    hit_meta = hit.get("metadata", {}) if hit else {}
    if hit_meta.get("task_template") == current_meta.get("task_template"):
        return "same_task_template"
    if hit_meta.get("app") and hit_meta.get("app") == current_meta.get("app"):
        return "same_app_different_task"
    return "different_app_or_unknown"


def app_for_task(task_template: str) -> str:
    for prefix, app in TASK_APP_HINTS.items():
        if task_template.startswith(prefix):
            return app
    return "unknown"


def summarize_run(episodes: List[Dict[str, Any]], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = summarize_metrics(episodes, steps)
    return {
        "created_at": utc_now(),
        "metrics": metrics,
        "by_task_template": summarize_group(episodes, steps, "task_template"),
        "by_app": summarize_group(episodes, steps, "app"),
        "cache_reuse_groups": summarize_step_hit_groups(steps),
    }


def summarize_metrics(episodes: List[Dict[str, Any]], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    episode_latencies = [float(item["episode_latency_seconds"]) for item in episodes if item.get("episode_latency_seconds") is not None]
    step_latencies = [float(item["step_latency_seconds"]) for item in steps if item.get("step_latency_seconds") is not None]
    inference_latencies = [float(item["inference_latency_seconds"]) for item in steps if item.get("inference_latency_seconds") is not None]
    cache_records = [item.get("cache") for item in steps if isinstance(item.get("cache"), dict)]
    lookup_latencies = [float(record.get("cache_lookup_seconds", 0.0)) for record in cache_records]
    hit_count = sum(1 for item in steps if item.get("cache_hit"))
    metrics = {
        "num_episodes": len(episodes),
        "num_steps": len(steps),
        "task_success_rate": mean_bool(item.get("success") for item in episodes),
        "cache_hit_rate": hit_count / len(steps) if steps else 0.0,
        "page_cache_hit_rate": mean_bool(record.get("page_cache_hit") for record in cache_records),
        "processor_cache_hit_rate": mean_bool(record.get("processor_cache_hit") for record in cache_records),
        "average_step_latency_seconds": statistics.fmean(step_latencies) if step_latencies else 0.0,
        "average_model_inference_latency_seconds": statistics.fmean(inference_latencies) if inference_latencies else 0.0,
        "average_episode_latency_seconds": statistics.fmean(episode_latencies) if episode_latencies else 0.0,
        "model_invocation_count": sum(1 for item in steps if item.get("model_invoked")),
        "cache_lookup_overhead_seconds": statistics.fmean(lookup_latencies) if lookup_latencies else 0.0,
        "cache_lookup_total_seconds": sum(lookup_latencies),
    }
    return metrics


def summarize_group(episodes: List[Dict[str, Any]], steps: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    values = sorted({str(item.get(key, "unknown")) for item in episodes + steps})
    return {
        value: summarize_metrics(
            [item for item in episodes if str(item.get(key, "unknown")) == value],
            [item for item in steps if str(item.get(key, "unknown")) == value],
        )
        for value in values
    }


def summarize_step_hit_groups(steps: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups = ("same_task_template", "same_app_different_task", "different_app_or_unknown", "miss")
    result = {}
    for group in groups:
        selected = [item for item in steps if item.get("cache_reuse_group") == group]
        result[group] = {
            "num_steps": len(selected),
            "hit_rate": sum(1 for item in selected if item.get("cache_hit")) / len(selected) if selected else 0.0,
        }
    result["overall"] = {
        "num_steps": len(steps),
        "hit_rate": sum(1 for item in steps if item.get("cache_hit")) / len(steps) if steps else 0.0,
    }
    return result


def mean_bool(values: Iterable[Any]) -> float:
    items = [bool(value) for value in values]
    return sum(1 for value in items if value) / len(items) if items else 0.0


def build_run_config(args: argparse.Namespace, defaults: BenchmarkDefaults, tasks: Sequence[str]) -> Dict[str, Any]:
    data = vars(args).copy()
    data["config"] = str(args.config)
    data["resolved_defaults"] = asdict(defaults)
    data["resolved_tasks"] = list(tasks)
    data["created_at"] = utc_now()
    return sanitize(data)


def append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(sanitize(value), ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(data), ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


if __name__ == "__main__":
    raise SystemExit(main())
