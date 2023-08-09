# EXAMPLE COMMANDS
# python decode_icl_llama.py --alignment_source oasstgpt --model_name meta-llama/Llama-2-13b-hf --output_fn llama2-13B_icl-oasstgpt_outputs_temp0.7.jsonl
# <DEBUG PRINT_ICL> # python decode_icl_llama.py --alignment_source oasstgpt --model_name meta-llama/Llama-2-13b-hf --output_fn llama2-13B_icl-oasstgpt_printicl.jsonl
# <DEBUG ENTER_CLI> # python decode_icl_llama.py --alignment_source oasstgpt --model_name meta-llama/Llama-2-13b-hf --output_fn llama2-13B_icl-oasstgpt_entercli.jsonl

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


r_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
r_model = AutoModel.from_pretrained('facebook/contriever')

# embedding computation helper
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

# Han: embeddings currently computed on CPU; each instance takes less than 0.2 sec so probably ok for now
def compute_embeddings(r_tokenizer, r_model, sentences):
    with torch.no_grad():
        # Apply tokenizer
        inputs = r_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        outputs = r_model(**inputs)
        embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings

def prepare_lima_data_and_index(r_tokenizer, r_model):
    lima_dataset = load_dataset("GAIR/lima")
    idx2alignment_data = dict()
    alignment_prompt_list = [e['conversations'][0] for e in lima_dataset['train']]
    alignment_data_prompt_embeddings = compute_embeddings(r_tokenizer, r_model, alignment_prompt_list)
    
    for i, e in enumerate(lima_dataset['train']):
        context = ""
        prompt = e['conversations'][0]
        response = e['conversations'][1]
        idx2alignment_data[i] = {'context': context.strip(), 'prompt': prompt.strip(), 'response': response.strip()}

    return (idx2alignment_data, alignment_data_prompt_embeddings)

def prepare_oasstgpt_data_and_index(r_tokenizer, r_model):
    oasstgpt_dataset = load_dataset('json', data_files="resources/oasst_gpt_9k.jsonl")
    idx2alignment_data = dict()
    alignment_prompt_list = [e['prompt'] for e in oasstgpt_dataset['train']]
    alignment_data_prompt_embeddings = compute_embeddings(r_tokenizer, r_model, alignment_prompt_list)
    
    for i, e in enumerate(oasstgpt_dataset['train']):
        context = ""
        prompt = e['prompt']
        response = e['final_return']
        idx2alignment_data[i] = {'context': context.strip(), 'prompt': prompt.strip(), 'response': response.strip()}

    return (idx2alignment_data, alignment_data_prompt_embeddings)

def dummy_prepare_oasstgpt_data_and_index(r_tokenizer, r_model):
    oasstgpt_dataset = load_dataset('json', data_files="resources/oasst_gpt_9k.jsonl")
    idx2alignment_data = dict()
    
    for i, e in enumerate(oasstgpt_dataset['train']):
        context = ""
        prompt = e['prompt']
        response = e['final_return']
        idx2alignment_data[i] = {'context': context.strip(), 'prompt': prompt.strip(), 'response': response.strip()}

    return (idx2alignment_data, len(oasstgpt_dataset['train']))

def top_k_alignment_idx(alignment_embeddings, prompt, k):
    if isinstance(alignment_embeddings, int): # HACK: special mode -- random selection
        len_alignment_embeddings = alignment_embeddings
        return random.sample(range(len_alignment_embeddings), k)
    
    prompt_embed = compute_embeddings(r_tokenizer, r_model, [prompt]) # Han: assuming bs=1 but can change later
    scores = alignment_embeddings.matmul(prompt_embed.T) # shape: #_of_alignment_database, bs_of_prompts
    sorted_idx = torch.argsort(scores, dim=0, descending=True)
    sorted_idx_list = sorted_idx[:k].squeeze(-1).tolist() # assuming bs=1
    return sorted_idx_list

def alignment_icl_retriever(prompt, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, icl_mode,
                            max_k=20, max_tokens=3000, max_per_instance_tokens=500):
    if icl_mode == "short": # for Llama-1
        max_k = 20
        max_tokens = 1000
        max_per_instance_tokens = 500
    elif icl_mode == "default": # for Llama-2
        pass
    else:
        raise ValueError(f"icl mode {icl_mode} not supported")
    sorted_idx_list = top_k_alignment_idx(alignment_data_prompt_embeddings, prompt, max_k)
    alignment_icl_data = []
    total_approx_len = 0
    for idx in sorted_idx_list:
        # Han: below doesn't need to be too accurate, just for max length estimation
        if idx2alignment_data[idx]['context']:
            raise ValueError("context in prompt not supported yet, need to update eos_id_list")
            approx_string = f"Context: {idx2alignment_data[idx]['context']}\n\nQuestion: {idx2alignment_data[idx]['prompt']}\n\nHere\u2019s an example answer: {idx2alignment_data[idx]['response']}\n\n"
        else:
            approx_string = f"Question: {idx2alignment_data[idx]['prompt']}\n\nHere\u2019s an example answer: {idx2alignment_data[idx]['response']}\n\n"
        len_approx_string = len(tokenizer(approx_string, add_special_tokens=False).input_ids)

        if len_approx_string > max_per_instance_tokens:
            continue
        total_approx_len += len_approx_string
        if total_approx_len <= max_tokens:
            alignment_icl_data.append(idx2alignment_data[idx])
        else:
            break
    return alignment_icl_data


def alignment_icl_wrapper(prompt, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, verbose=False, icl_mode="default"):
    template_list = []
    # Han: put most relevant demo example in the end, can change the strategy later
    selected_icl_data = list(reversed(alignment_icl_retriever(prompt, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, icl_mode)))
    
    if verbose:
        print('\n', "#" * 16)
        print(selected_icl_data)
        print("#" * 16, '\n')
    
    for e in selected_icl_data:
        if e['context']:
            template_list.append(f"Context: {e['context'].strip()}\n\nQuestion: {e['prompt'].strip()}\n\nHere\u2019s an example answer: {e['response'].strip()}")
        else:
            template_list.append(f"Question: {e['prompt'].strip()}\n\nHere\u2019s an example answer: {e['response'].strip()}")
    template_list.append(f"Question: {prompt}\n\nHere\u2019s an example answer:")
    template = '\n\n'.join(template_list)
    eos_string = "\nQuestion:" # Han: allow the model to skip one newline char
    tested_eos_id_list = tokenizer(eos_string, add_special_tokens=False).input_ids[1:] # Han: tokenizer-specific, try with the tokenizer first for new models!!!
    return {'string': template, 'list_of_strings': template_list, 'eos_id_list': tested_eos_id_list}


def decode_wrapper(prompt, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, model, max_new_tokens=1000, temperature=0.7, top_p=1.0, icl_mode="default"):
    wrapper_return = alignment_icl_wrapper(prompt, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, icl_mode=icl_mode)
    formatted_prompt, eos_id_list = wrapper_return['string'], wrapper_return['eos_id_list']

    total_input_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(inputs=total_input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=True)
        output_ids = outputs[0][len(total_input_ids[0]):] # assuming bs=1

    # Han: post-hoc truncation, this should be done ad-hoc in real use [TODO]
    output_ids = output_ids.tolist()
    end_idx = None
    for _i, _e in enumerate(output_ids):
        if output_ids[max((_i+1) - len(eos_id_list), 0):(_i+1)] == eos_id_list:
            end_idx = (_i+1) - len(eos_id_list)
            break
    processed_string = tokenizer.decode(output_ids[:end_idx], skip_special_tokens=True).strip()
    
    return processed_string


def print_icl_in_decode_wrapper(prompt, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, model, max_new_tokens=1000, temperature=0.7, top_p=1.0, icl_mode="default"):
    wrapper_return = alignment_icl_wrapper(prompt, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, icl_mode=icl_mode)
    formatted_prompt, eos_id_list = wrapper_return['string'], wrapper_return['eos_id_list']
    return formatted_prompt


def main(
    alignment_source: str,
    model_name: str,
    output_fn: str,
):
    lima_dataset = load_dataset("GAIR/lima") # using LIMA test

    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    if alignment_source == "lima":
        idx2alignment_data, alignment_data_prompt_embeddings = prepare_lima_data_and_index(r_tokenizer, r_model)
    elif alignment_source == "oasstgpt":
        idx2alignment_data, alignment_data_prompt_embeddings = prepare_oasstgpt_data_and_index(r_tokenizer, r_model)
    elif alignment_source == "random_oasstgpt":
        idx2alignment_data, alignment_data_prompt_embeddings = dummy_prepare_oasstgpt_data_and_index(r_tokenizer, r_model)
    else:
        raise ValueError(f"alignment source {alignment_source} not supported")

    if 'huggyllama' in model_name: # hard-coded for Llama-1 now
        icl_mode = "short"
    else:
        icl_mode = "default"

    if 'printicl' in output_fn: # hard-coded for printing mode now
        # Han: only for debugging
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = None # no need for an actual model for printing
        generations = []
        prompt_list = [e['conversations'][0] for e in lima_dataset['test']]
        # repeat = 1
        start_time = time.time()
        for i, prompt in enumerate(prompt_list):
            generations.append({'prompt': prompt, 'icl_text': print_icl_in_decode_wrapper(prompt, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, model, icl_mode=icl_mode)})
            if (i+1) % 10 == 0:
                print(f"finished {i+1} prompts in {time.time() - start_time} seconds")
    elif 'entercli' in output_fn:
        # Han: only for debugging
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto")
        generations = []
        do_loop = True
        while do_loop:
            breakpoint()
            # print([decode_wrapper(testp, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, model, icl_mode=icl_mode)])
            # print([print_icl_in_decode_wrapper(testp, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, model, icl_mode=icl_mode)])
            # with open(output_fn, "a") as f: f.write(json.dumps({'prompt': testp, 'answer': [decode_wrapper(testp, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, model, icl_mode=icl_mode)], 'context': [print_icl_in_decode_wrapper(testp, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, model, icl_mode=icl_mode)]}))
    else:
        # Han: this is the generation mode
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto")
        generations = []
        prompt_list = [e['conversations'][0] for e in lima_dataset['test']]
        repeat = 5
        start_time = time.time()
        for i, prompt in enumerate(prompt_list):
            generations.append({'prompt': prompt, 'response_list': [decode_wrapper(prompt, idx2alignment_data, alignment_data_prompt_embeddings, tokenizer, model, icl_mode=icl_mode) for _ in range(repeat)]})
            if (i+1) % 10 == 0:
                print(f"finished {i+1} prompts in {time.time() - start_time} seconds")

    with open(output_fn, "w") as f:
        for gen in generations:
            f.write(json.dumps(gen))
            f.write('\n')


if __name__ == "__main__":
    fire.Fire(main)