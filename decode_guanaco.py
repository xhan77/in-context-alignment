# EXAMPLE COMMANDS
# python decode_guanaco.py --model_name huggyllama/llama-7b --adapter_name timdettmers/guanaco-7b --output_fn guanaco-7B_outputs_temp0.7.jsonl

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


def bsl_decode_wrapper(prompt, bsl_tokenizer, bsl_model, max_new_tokens=1000, temperature=0.7, top_p=1.0):
    formatted_prompt = (
        f"A chat between a curious human and an artificial intelligence assistant."
        f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        f"### Human: {prompt} ### Assistant:"
    ) # guanaco-specific format
    with torch.no_grad():
        inputs = bsl_tokenizer(formatted_prompt, return_tensors="pt").to(bsl_model.device)
        outputs = bsl_model.generate(inputs=inputs.input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=True)
        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()

        end_idx = len(output_ids)
        processed_string = bsl_tokenizer.decode(output_ids[:end_idx], skip_special_tokens=True).strip()
        
    return processed_string


def main(
    model_name: str,
    adapter_name: str,
    output_fn: str,
):
    lima_dataset = load_dataset("GAIR/lima") # using LIMA test

    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    bsl_model_name = model_name
    bsl_adapters_name = adapter_name
    bsl_model = AutoModelForCausalLM.from_pretrained(
        bsl_model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    bsl_model = PeftModel.from_pretrained(bsl_model, bsl_adapters_name)
    bsl_tokenizer = AutoTokenizer.from_pretrained(bsl_model_name)

    generations = []
    prompt_list = [e['conversations'][0] for e in lima_dataset['test']]
    repeat = 5
    start_time = time.time()
    for i, prompt in enumerate(prompt_list):
        generations.append({'prompt': prompt, 'response_list': [bsl_decode_wrapper(prompt, bsl_tokenizer, bsl_model) for _ in range(repeat)]})
        if (i+1) % 10 == 0:
            print(f"finished {i+1} prompts in {time.time() - start_time} seconds")

    with open(output_fn, "w") as f:
        for gen in generations:
            f.write(json.dumps(gen))
            f.write('\n')


if __name__ == "__main__":
    fire.Fire(main)