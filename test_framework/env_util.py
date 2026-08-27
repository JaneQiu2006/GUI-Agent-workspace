import json
import os
import time
import signal


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
        
def generate_step_list(SOP):
    step_list = []
    for _, steps in SOP:
        step_list.extend(steps.split('>'))  
    step_list = [step.replace("页面", "") for step in step_list]
    
    return step_list


def batch_interact_environment(agent, env, ocr_detection, ocr_recognition,\
        accelerator, post_f = lambda x: x, use_tqdm = True, decode_f = lambda x: x, task_id=0, config=None):
    config = config or {}

    try:
        env.terminated = False
        reset_success = False
        env.image_id = str(time.time())
        env.steps = 0
        while not (reset_success):
            for _ in range(5):
                try:
                    if accelerator.is_main_process:
                        with timeout(seconds=480): # change this if frequently timeout
                            env.task_id=task_id
                            #print(env.task_id)
                            obs = env.get_obs()
                            #print('----------------------')
                            #print(obs)
                        reset_success = True
                    break
                except Exception as e:
                    print(f"Error in environment reset")
                    print(e)
                    continue
        done = False
        
        steps = 0
        obs['previous_actions'] = []
        obs['image_paths'] = []
        

        db = None
        embedding_model = None
        if config.get("use_rag", True):
            from RAGToolbox import Jinaembedding, Vectordatabase

            script_dir = os.path.dirname(os.path.abspath(__file__))
            database_path = config.get(
                "rag_database_path", os.path.join(script_dir, "rag_database")
            )
            embedding_model = Jinaembedding(path=config["embedding_model"])
            db = Vectordatabase()
            db.load_vector(database_path)

        while not (done):
            
            steps += 1
            if accelerator.is_main_process:
                print(f"Environment steps {str(steps)}")
                print("getting actions!")
                
                sop_start_time = time.time()
                SOP = db.query_score(obs['task'], embedding_model, 1) if db else []
                sop_end_time = time.time()
                print("sop time:",sop_end_time-sop_start_time)
                
                obs['low-level']= generate_step_list(SOP)

                start_time = time.time()                
                res = agent.get_action(obs)
                end_time = time.time()
                print("agent time:",end_time-start_time)           
                obs['action'] = res['action']            
                obs['success'] = False
                obs['input_token_count'] = res['input_token_count']
                obs['output_token_count'] = res['output_token_count']
                action = obs['action']
                step_start_time = time.time()
                with timeout(seconds=5*60):
                    step_return = env.step(decode_f(action))
                obs_dict, terminate, success = step_return
                step_end_time = time.time()             
                print("step time:",step_end_time-step_start_time)

                if obs['action'] == "COMPLETE":
                    success = True
                    terminate = True

                
                
                if success:
                    obs['success'] = True
                
                file_path = os.path.join(config['save_path'], config['json_name'])

                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        try:
                            data = json.load(file)
                        except json.JSONDecodeError:
                            data = [] 
                else:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    data = []
                
            
                time_difference = end_time - start_time
                #print(time_difference)
                obs['latency'] = time_difference

                
                data.append(obs)
                print(obs)
                

                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=4, ensure_ascii=False) 
                                             
                if terminate:
                    if success:
                        return 1
                    else:
                        return 0

                obs['previous_actions'].append(obs['action'])
                obs['image_paths'].append(obs['image_path'])   
                obs["image_path"] = obs_dict["image_path"]
        
        return 0           
            
    except Exception as e:
        print(f"Error in environment interaction")
        import traceback
        print(traceback.format_exc())
        print(e)
        env.terminate()
        return 0

                
        
