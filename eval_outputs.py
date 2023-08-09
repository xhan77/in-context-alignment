# EXAMPLE COMMANDS
# python eval_outputs.py --model_outputs_fn "guanaco-7B_outputs_temp0.7.jsonl" --bsl_outputs_fn "davinci003_results.jsonl" --eval_out_fn "guanaco-7B_vs_davinci003_autoeval.jsonl"

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

openai.api_key = "TODO" # SECRET
openai.organization = "TODO" # SECRET

mode = "default" # "default" for 1500 comparisons, "short" for 300 comparisons with a lower budget


def generate_auto_eval_text(e1, e2, print_text=True):
    answer_1 = e1['response']
    answer_2 = e2['response']
    instruction = e1['prompt']
    instruction2 = e2['prompt']
    assert instruction == instruction2
    
    eval_prompt = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    auto_eval_text = f"[Question]\n{instruction}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{eval_prompt}\n\n"
    
    if print_text:
        print(auto_eval_text)
    
    return auto_eval_text

def query_openai_autoeval(query_text): # Han: beware of the billing quota!
    response = None
    prompt = query_text

    num_trials = 5
    while response is None and num_trials > 0:
        try:
            response = openai.ChatCompletion.create(
              model="gpt-4-0613",
              messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {"role": "user", "content": prompt}
              ],
              temperature=0.2,
              max_tokens=500,
            )
        except Exception as query_err:
            print(f"query failure, will try again: {query_err}")
            time.sleep(3)
            num_trials -= 1
            continue

    return response.choices[0].message['content']

def gpt4_eval(model_results_fn, bsl_model_results_fn, mode="default"):
    model_results_dataset = load_dataset('json', data_files=model_results_fn, split='train')
    bsl_model_results_dataset = load_dataset('json', data_files=bsl_model_results_fn, split='train')
    openai_return_list = []
    i = 0
    start_time = time.time()
    for e1, e2 in zip(model_results_dataset, bsl_model_results_dataset):
        assert 'response_list' not in e2
        openai_return_list.append([])
        if 'response_list' in e1:
            if mode == "default":
                iterlist = e1['response_list']
            elif mode == "short":
                iterlist = e1['response_list'][:1]
            else:
                raise ValueError("invalid mode")
            for res in iterlist:
                openai_return_list[-1].append(query_openai_autoeval(generate_auto_eval_text({'prompt': e1['prompt'], 'response': res}, e2, print_text=False)))
        else:
            raise ValueError("disabled for now")
            openai_return_list[-1].append(query_openai_autoeval(generate_auto_eval_text(e1, e2, print_text=False)))
        
        i += 1
        if i % 10 == 0:
            print(f"finished {i} prompts in {time.time() - start_time} seconds")

    return openai_return_list


def main(
    model_outputs_fn: str,
    bsl_outputs_fn: str,
    eval_out_fn: str,
):
    model_results_fn = model_outputs_fn
    bsl_model_results_fn = bsl_outputs_fn
    autoeval_fn = eval_out_fn

    openai_return_list = gpt4_eval(model_results_fn, bsl_model_results_fn, mode=mode)
    with open(autoeval_fn, "w") as f:
        for gen_list in openai_return_list:
            f.write(json.dumps({'autoeval_text_list': gen_list}))
            f.write('\n')

    # raw analysis, use the exported file for detailed analyses
    print("query to openai completed, now analyzing results ...")
    openai_return_list = []
    for e in load_dataset('json', data_files=autoeval_fn, split='train'):
        openai_return_list.extend(e['autoeval_text_list'])

    win_for_1 = 0
    win_for_2 = 0
    draw = 0
    avg_1 = []
    avg_2 = []
    for idx in range(len(openai_return_list)):
        score_1 = float(openai_return_list[idx].split('\n')[0].split(' ')[0]) # Han: no parsing errors so far
        score_2 = float(openai_return_list[idx].split('\n')[0].split(' ')[1]) # Han: no parsing errors so far
        avg_1.append(score_1)
        avg_2.append(score_2)
        if score_1 > score_2:
            win_for_1 += 1
        elif score_1 < score_2:
            win_for_2 += 1
        else:
            draw += 1
    print("win/draw/lose:", win_for_1, draw, win_for_2)
    print("avg score model/bsl:", np.mean(avg_1), np.mean(avg_2))
    print("filename:", eval_out_fn)


if __name__ == "__main__":
    fire.Fire(main)