#!/usr/bin/env python3
import os
import argparse
from utils.model import Model
from utils.reward_model import reward_factory
from utils.load_data import load_dataset, save_json_data, load_json_data, load_remain_data, load_rm_dataset, extract_answer, get_final_pred
from tqdm import tqdm
from prompt.main_exp_prompt import system_prompt_dic
import random 

random.seed(17)

def rm_infer(args):
    remote = False if args.url == '139' else True
    reward_model = reward_factory('Skywork', remote)
    result_path = f"./result/main_exp/all/{args.model}_{args.method}.json"
    response_dic = {item['id']: item['response'] for item in load_json_data(result_path)}
    dataset = load_rm_dataset( n_examples=args.n_example,
                           shuffle=args.shuffle)
    output_path = f"./result/rm/{args.model}_skywork.json"
    results, dataset = load_remain_data(output_path, dataset)
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Making inference"):
        response = response_dic[item['id']]
        best_answer, best_response, scores = reward_model.find_bestn_answer(item['input'], response)
        msg = {
            'id': item['id'],
            'img_path': item['img_path'],
            'question': item['question'],
            'response': response,
            'preds': extract_answer(response),
            'score': scores,
            'best_response': best_response,
            'pred_answer': best_answer,
            'golden_answer': item['golden_answer'],
            'correct': item['golden_answer'] == best_answer
        }
        results.append(msg)
        
        save_json_data(output_path, results)

    print(f"\nTesting complete. Updated dataset saved to {output_path}")


def cal_usage(args):
    model = Model(args.model, args.url, args.port)
    dataset = load_dataset(system_prompt=system_prompt_dic[args.method], 
                           n_examples=args.n_example,
                           shuffle=args.shuffle,
                           mini=True)
    generation_config = {
        'temperature':args.temperature,
        'n':args.n_sample,
        'top_p':args.top_p,
        'max_tokens':args.max_tokens,
        'seed':17,
        'usage':True
    }

    output_dir = f'./result/usage/' 
    output_path = os.path.join(output_dir, f"{args.model}_{args.method}.json")
    
    # results, start_idx = load_last_index(output_path)
    results, dataset = load_remain_data(output_path, dataset)
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Making inference"):
        response, usage = model.generate(message=item['input'], generation_config=generation_config)
        msg = {
            'id': item['id'],
            'img_path': item['img_path'],
            'question': item['question'],
            'response': response,
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'total_tokens': usage.total_tokens
        }
        results.append(msg)
    # Save the updated dataset
        save_json_data(output_path, results)

    print(f"\nTesting complete. Updated dataset saved to {output_path}")
  
    
def main_exp(args):
    model = Model(args.model, args.url, args.port)
    dataset = load_dataset(system_prompt=system_prompt_dic[args.method], 
                           n_examples=args.n_example,
                           shuffle=args.shuffle)
    generation_config = {
        'temperature':args.temperature,
        'n':args.n_sample,
        'top_p':args.top_p,
        'max_tokens':args.max_tokens,
        'seed':17
    }
    
    if args.option == 'main_exp':
        output_dir = f'./result/main_exp/all/'
    elif args.option == 'budget':
        output_dir = f'./result/budget/'
    elif args.option == 'n_scale':
        output_dir = f'./result/n_scale/' 
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.option == 'main_exp':
        output_path = os.path.join(output_dir, f"{args.model}_{args.method}.json")
    elif args.option == 'budget':
        output_path = os.path.join(output_dir, f"{args.model}.json")
    elif args.option == 'n_scale':
        output_path = os.path.join(output_dir, f"{args.model}_{args.n_sample}.json")
    
    # results, start_idx = load_last_index(output_path)
    results, dataset = load_remain_data(output_path, dataset)
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Making inference"):
        response = model.generate(message=item['input'], generation_config=generation_config)
        preds = extract_answer(response_text=response)
        pred_answer = get_final_pred(preds) if preds else None
        golden_answer = item['golden_answer']
        cor_flag = [pred == golden_answer for pred in preds]
        correct = pred_answer == golden_answer
        msg = {
            'id': item['id'],
            'img_path': item['img_path'],
            'question': item['question'],
            'response': response,
            'preds': preds,
            'pred_answer': pred_answer,
            'golden_answer': golden_answer,
            'cor_flag': cor_flag,
            'correct': correct
        }
        results.append(msg)
    # Save the updated dataset
        save_json_data(output_path, results)

    print(f"\nTesting complete. Updated dataset saved to {output_path}")

def main(args):
    if args.option == 'rm':
        rm_infer(args)
    elif args.option == 'usage':
        cal_usage(args)
    else:
        main_exp(args)
    # Load the dataset
   

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='cot')
parser.add_argument('--n_example', type=int, default=3000)
parser.add_argument('--n_sample', type=int, default=8)
parser.add_argument('--model', type=str, default='Qwen2.5-VL-7B')
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--top_p', type=float, default=0.5)
parser.add_argument('--max_tokens', type=int, default=5000)
parser.add_argument('--url', type=str, default='120')
parser.add_argument('--port', type=str, default='14396')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--option', type=str, default='main_exp')
args = parser.parse_args()
if __name__ == "__main__":
    main(args)

