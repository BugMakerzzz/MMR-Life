#!/usr/bin/env python3

import os
import argparse
from utils.model import Model
from utils.load_data import load_oe_dataset, save_json_data, load_last_index
from tqdm import tqdm
from prompt.generate_neg_prompt import system_prompt_dic


def main(args):
    task_dir = f'./raw_question/{args.type}/{args.task}/'
    if args.mask:
        output_path = os.path.join(task_dir, f"neg_option_{args.model}_mask.json")
    else:
        output_path = os.path.join(task_dir, f"neg_option_{args.model}.json")
    # Load the dataset
    model = Model(args.model, args.url)
    dataset = load_oe_dataset(data_path=task_dir, system_prompt=system_prompt_dic[args.type][args.task], n_examples=args.n_example, masked=args.mask)
    generation_config = {
        'temperature':args.temperature,
        'n':args.neg_num,
        'max_tokens':args.max_tokens,
        'top_p':1.0,
        'seed':17
    }
    results, start_idx = load_last_index(output_path)
    for i, item in tqdm(enumerate(dataset[start_idx:], start=start_idx), total=len(dataset)-start_idx):
        response = model.generate(message=item['input'], generation_config=generation_config)
        msg = {
            'index': i+1,
            'source': item['source'],
            'question': item['question'],
            'golden_answer': item['golden_answer'],
            'wrong_response': response
        }
        results.append(msg)
    # Save the updated dataset
        save_json_data(output_path, results)
    print(f"\nTesting complete. Updated dataset saved to {output_path}")

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='abductive')
parser.add_argument('--task', type=str, default='cartoon')
parser.add_argument('--n_example', type=int, default=200)
parser.add_argument('--model', type=str, default='gpt-5-mini')
parser.add_argument('--url', type=str, default='154')
parser.add_argument('--neg_num', type=int, default=8)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--max_tokens', type=int, default=5000)
parser.add_argument('--mask', action='store_true')
args = parser.parse_args()
if __name__ == "__main__":
    main(args)

