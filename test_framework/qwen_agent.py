import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
#from qwen_vl_utils import process_vision_info
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
import requests
import json
import re
import PIL.Image
import math
import yaml
import re
from awq import AutoAWQForCausalLM
from awq.utils.qwen_vl_utils import process_vision_info
from phone_prompt import build_phone_prompt

class QwenAgent:
    def __init__(self, device, accelerator, cache_dir='~/.cache', dropout=0.5, policy_lm=None,
                 max_new_tokens=32, use_bfloat16=False):
        # 加载模型和处理器
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                policy_lm,  torch_dtype=torch.float16, device_map="balanced", attn_implementation="flash_attention_2",
            ).to(device)
        except Exception:
            print("disable flash attention")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                policy_lm,  torch_dtype=torch.float16, device_map="balanced", 
            ).to(device)
        self.processor = AutoProcessor.from_pretrained(policy_lm)
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
  
    def prepare(self): 
        self.model = self.accelerator.prepare(self.model)

    def _get_a_action(self, obs):
        sys_prompt = build_phone_prompt(
            obs['task'], obs.get('previous_actions'), obs.get('low-level')
        )
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": sys_prompt,
                            },
                            {
                                "type": "image",
                                "image": obs['image_path'],
                            },
                        ],
                    }
                ]

        # 处理输入并生成
        chat_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
                    text=[chat_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
        self.device = self.model.device
        inputs = inputs.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128).to(self.device)
        generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
        output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

        print(output_text[0])
        action = output_text[0].strip()
        print(action)
        #    华为
        

        #prefix = 'actions:\n'
        #start_index = output_text[0].find(prefix) + len(prefix)
        #action = output_text[0][start_index:]


        
        input_tokens = self.processor.tokenizer.tokenize(chat_text)
        input_token_count = len(input_tokens)
        output_tokens = self.processor.tokenizer.tokenize(output_text[0])
        output_token_count = len(output_tokens)
        return action, input_token_count, output_token_count


    def get_action(self, obs):
        result = {}
        osatlas_action, input_token_count, output_token_count = self._get_a_action(obs)
        result['action'] = osatlas_action
        result['input_token_count'] = input_token_count
        result['output_token_count'] = output_token_count
        return result



class QwenAwqAgent:
    def __init__(self, device, accelerator, cache_dir='~/.cache', dropout=0.5, policy_lm=None,
                 max_new_tokens=32, use_bfloat16=False):
        # 加载模型和处理器
        try:
            self.model = AutoAWQForCausalLM.from_quantized(
                policy_lm, device_map="auto", attn_implementation="flash_attention_2",
            )
        except Exception:
            print("disable flash attention")
            self.model =  AutoAWQForCausalLM.from_quantized(
                policy_lm,   device_map="auto", 
            )
        self.processor = AutoProcessor.from_pretrained(policy_lm)
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
  
    def prepare(self): 
        self.model = self.accelerator.prepare(self.model)

    def _get_a_action(self, obs):
        sys_prompt = build_phone_prompt(
            obs['task'], obs.get('previous_actions'), obs.get('low-level')
        )
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": sys_prompt,
                            },
                            {
                                "type": "image",
                                "image": obs['image_path'],
                            },
                        ],
                    }
                ]

        # 处理输入并生成
        chat_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
                    text=[chat_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
        self.device = self.model.device
        inputs = inputs.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128).to(self.device)
        generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
        output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

        print(output_text[0])
        action = output_text[0].strip()
        print(action)
        #    华为
        

        #prefix = 'actions:\n'
        #start_index = output_text[0].find(prefix) + len(prefix)
        #action = output_text[0][start_index:]


        
        input_tokens = self.processor.tokenizer.tokenize(chat_text)
        input_token_count = len(input_tokens)
        output_tokens = self.processor.tokenizer.tokenize(output_text[0])
        output_token_count = len(output_tokens)
        return action, input_token_count, output_token_count


    def get_action(self, obs):
        result = {}
        osatlas_action, input_token_count, output_token_count = self._get_a_action(obs)
        result['action'] = osatlas_action
        result['input_token_count'] = input_token_count
        result['output_token_count'] = output_token_count
        return result
