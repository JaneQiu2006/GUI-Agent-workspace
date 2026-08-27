from env_util import batch_interact_environment
def the_entire_trajectory_loop(env,\
                agent,\
                accelerator,\
                tokenizer,\
                ocr_detection,\
                ocr_recognition,\
                eval_nums: int = 10,\
                use_wandb: bool = False,\
                save_path: str = None,\
                decode_f: callable = lambda x: x,\
                **kwargs):
    agent.prepare()
    runtime_config = dict(kwargs)
    runtime_config["save_path"] = save_path
    done_nums = 0
    for i in range(eval_nums):
        done_nums = done_nums + batch_interact_environment(agent = agent,\
                                                       env = env,\
                                                       accelerator = accelerator,\
                                                       use_tqdm=False,\
                                                       decode_f = decode_f,\
                                                       ocr_detection=ocr_detection,\
                                                       ocr_recognition=ocr_recognition,\
                                                       config=runtime_config,\
                                                       task_id=i)

    
    successful_rate = done_nums / eval_nums
    print("successful_rate:")
    print(successful_rate)
