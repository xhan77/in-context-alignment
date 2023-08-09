# EXAMPLE COMMANDS
# python decode_llamabase.py --model_name meta-llama/Llama-2-13b-hf --template zero-shot --output_fn llama2base-13B_zeroshot_outputs_temp0.7.jsonl

import requests
import pysbd
import json
import time
from datasets import load_dataset
from collections import Counter, defaultdict
import random
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
import openai
import fire


def bsl_decode_wrapper(prompt, demo_mode, bsl_tokenizer, bsl_model, max_new_tokens=1000, temperature=0.7, top_p=1.0):
    if demo_mode == "zero-shot":
        formatted_prompt = f"Question: {prompt.strip()}\n\nAnswer:"
    else:
        raise ValueError(f"demo mode {demo_mode} not supported")
    
    with torch.no_grad():
        inputs = bsl_tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(bsl_model.device)
        outputs = bsl_model.generate(inputs=inputs.input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=True)
        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()

        # end_idx = len(output_ids)
        eos_string = "\nQuestion:" # Han: sometimes the model would skip one newline char
        eos_id_list = bsl_tokenizer(eos_string, add_special_tokens=False).input_ids[1:]
        end_idx = None
        for _i, _e in enumerate(output_ids):
            if output_ids[max((_i+1) - len(eos_id_list), 0):(_i+1)] == eos_id_list:
                end_idx = (_i+1) - len(eos_id_list)
                break

        processed_string = bsl_tokenizer.decode(output_ids[:end_idx], skip_special_tokens=True).strip()
        
    return processed_string


def main(
    model_name: str,
    template: str,
    output_fn: str,
):
    lima_dataset = load_dataset("GAIR/lima") # using LIMA test

    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    bsl_model_name = model_name
    bsl_tokenizer = AutoTokenizer.from_pretrained(bsl_model_name)
    bsl_model = AutoModelForCausalLM.from_pretrained(bsl_model_name, torch_dtype=torch_dtype, device_map="auto")

    generations = []
    prompt_list = [e['conversations'][0] for e in lima_dataset['test']]
    repeat = 5
    start_time = time.time()
    for i, prompt in enumerate(prompt_list):
        generations.append({'prompt': prompt, 'response_list': [bsl_decode_wrapper(prompt, template, bsl_tokenizer, bsl_model) for _ in range(repeat)]})
        if (i+1) % 10 == 0:
            print(f"finished {i+1} prompts in {time.time() - start_time} seconds")

    with open(output_fn, "w") as f:
        for gen in generations:
            f.write(json.dumps(gen))
            f.write('\n')


if __name__ == "__main__":
    fire.Fire(main)