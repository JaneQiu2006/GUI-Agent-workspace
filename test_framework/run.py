import json
import os


def load_task_file(assets_path):
    all_tasks = []
    with open(os.path.join(assets_path, "instructions.txt")) as fb: 
        for line in fb:
            all_tasks.append(line.replace("\n", ""))
    return all_tasks


def load_config(path="./config/config.yaml"):
    """Load JSON-compatible YAML without requiring PyYAML for JSON configs."""
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "配置不是 JSON 格式；读取传统 YAML 配置需要安装 PyYAML"
            ) from exc
        return yaml.safe_load(content)


def run_full_evaluation():
    from env import AndroidEmulator
    from loop import the_entire_trajectory_loop
    from action_util import qwen_translate_action

    config = load_config()
    all_tasks = load_task_file(config['assets_path'])
    translate_action = qwen_translate_action
    decode_f = lambda x: x

    import torch
    import transformers
    from accelerate import Accelerator
    from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
    from datetime import timedelta
    from qwen_agent import QwenAgent

    transformers.logging.set_verbosity_error()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        InitProcessGroupKwargs(timeout=timedelta(minutes=40)),
        kwargs_handlers=[ddp_kwargs],
        project_dir=config['save_path'],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = QwenAgent(
        device=device,
        accelerator=accelerator,
        policy_lm=config['policy_lm'],
        max_new_tokens=config['max_new_tokens'],
    )


    def construct_env(sample_mode):
        env = AndroidEmulator(
            max_steps=config['max_steps']-1, 
            translate_action=translate_action,
            temp_path = os.path.join(config['save_path'], "images"),
            save_images=True,
            all_tasks=all_tasks,
            sample_mode=sample_mode,
            task_id=0,
            device_config=config.get('device', {}),
        )
        return env


    env = construct_env(sample_mode=config['eval_sample_mode'])
    the_entire_trajectory_loop(env = env,
                tokenizer=agent.tokenizer,
                agent = agent,
                accelerator = accelerator,
                decode_f=decode_f,
                ocr_detection=None,
                ocr_recognition=None,
                **config)                       


def main():
    run_full_evaluation()

if __name__ == "__main__":
    main()
