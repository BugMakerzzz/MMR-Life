from utils.load_data import load_json_data, save_json_data, extract_answer, get_final_pred
from utils.config import TYPE_TASK_MAP, TASK_PATH_MAP
from utils.draw_fig import prepare_fig_input, draw_bar, draw_scatter, draw_line, draw_heat, draw_dendrogram
import argparse
import random 
random.seed(17)
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import tiktoken

TYPE_SPLIT_DICT = {
    'all': [4, 7], 
    'abductive': [4, 5], 
    'temporal': [5, 7], 
    'inductive': [3, 4], 
    'causal': [4, 6], 
    'deductive': [4, 7], 
    'analogical': [3, 4], 
    'spatial': [4, 6]
}

MODEL_URL = {
    # OpenAI (share the same OpenAI reference URLs)
    "gpt-5": "https://openai.com/index/introducing-gpt-5/",
    "gpt-5-mini": "https://openai.com/index/introducing-gpt-5/",
    "gpt-4.1": "https://openai.com/index/gpt-4-1/",
    "gpt-4.1-mini": "https://openai.com/index/gpt-4-1/",
    "gpt-4o": "https://openai.com/index/hello-gpt-4o/",
    "gpt-4o-mini": "https://openai.com/index/hello-gpt-4o/",
    "o4-mini": "https://openai.com/index/o3-o4-mini-system-card/",

    # Anthropic
    "claude-sonnet-4": "https://www.anthropic.com/news/claude-4",
    "claude-sonnet-4-thinking": "https://www.anthropic.com/news/claude-4",
    "claude-3.7-sonnet": "https://www.anthropic.com/news/claude-3-7-sonnet",

    # Google (Gemini 2.5 tech report on arXiv)
    "gemini-2.5-flash": "https://doi.org/10.48550/arXiv.2507.06261",
    "gemini-2.5-pro": "https://doi.org/10.48550/arXiv.2507.06261",

    # ByteDance (Doubao)
    "doubao-1.5-vision": "https://seed.bytedance.com/en/special/doubao_1_5_pro",

    # Open-source models (URLs are the cited technical reports/papers)
    "Kimi-VL-A3B": "https://arxiv.org/abs/2504.07491",
    "Kimi-VL-A3B-Thinking-2506": "https://arxiv.org/abs/2504.07491",
    "Keye-VL-1.5-8B": "https://arxiv.org/abs/2507.01949",
    "MiMo-VL-7B-RL": "https://arxiv.org/abs/2506.03569",
    "MiMo-VL-7B-SFT": "https://arxiv.org/abs/2506.03569",
    "MM-Eureka-Qwen-7B": "https://arxiv.org/abs/2503.07365",
    "MM-Eureka-Qwen-32B": "https://arxiv.org/abs/2503.07365",
    "OpenVLThinker-7B-v1.2": "https://arxiv.org/abs/2503.17352",
    "OpenVLThinker-7B-v1.2-sft": "https://arxiv.org/abs/2503.17352",
    "Qwen2.5-VL-7B": "https://arxiv.org/abs/2502.13923",
    "Qwen2.5-VL-32B": "https://arxiv.org/abs/2502.13923",
    "Qwen2.5-VL-72B": "https://arxiv.org/abs/2502.13923",
    "R1-Onevision-7B": "https://arxiv.org/abs/2503.10615",
    "R1-Onevision-7B-RL": "https://arxiv.org/abs/2503.10615",
    "Skywork-R1V-38B": "https://arxiv.org/abs/2504.05599",
    "VL-Rethinker-7B": "https://arxiv.org/abs/2504.08837",
    "VL-Rethinker-32B": "https://arxiv.org/abs/2504.08837",
    "VL-Rethinker-72B": "https://arxiv.org/abs/2504.08837",
    
    "InternVL3_5-8B": "https://arxiv.org/abs/2508.18265",
    "InternVL3_5-30B-A3B": "https://arxiv.org/abs/2508.18265",
    "InternVL3_5-38B": "https://arxiv.org/abs/2508.18265",
    "Gemma3-4B": "https://arxiv.org/abs/2503.19786",
    "Gemma3-12B": "https://arxiv.org/abs/2503.19786",
    "Gemma3-27B": "https://arxiv.org/abs/2503.19786",

    # Qwen (QVQ)
    "QVQ-72B-Preview": "https://qwenlm.github.io/blog/qvq-72b-preview/",
}

def get_leaderboard_results(results):
    table = []
    length = len(results['Model'])
    for idx in range(length):
        if not results['url'][idx]:
            print(results['Model'][idx])
            continue
        out = {}
        out['Model'] = results['Model'][idx]
        out['Source'] = results['url'][idx]
        out['Overall'] = results['all'][idx]
        for type in TYPE_TASK_MAP.keys():
            out[type.capitalize()] = results[type][idx]

        table.append(out)


    # remove the element
    human = {'Model': 'Human',
             'Source': '',
             'Overall': 72.28,
             'Abductive': 79.76,
             'Analogical': 57.65,
             'Causal': 75.00,
             'Deductive': 70.59 ,
             'Inductive': 63.41,
             'Spatial': 79.76,
             'Temporal': 79.76
        }
    human = {"-": human}

    # sort the table
    sorted_table = sorted(table, key=lambda x: x['Overall'], reverse=True)
    sorted_table = {str(i+1): sorted_table[i] for i in range(len(sorted_table))}

    # add the human performance back
    score_table = {**human, **sorted_table}


    # rename the top 3 models by adding 🥇, 🥈, and 🥉, respectively
    score_table['1']['Model'] = score_table['1']['Model'] + ' 🥇'
    score_table['2']['Model'] = score_table['2']['Model'] + ' 🥈'
    score_table['3']['Model'] = score_table['3']['Model'] + ' 🥉'

    # print to file
    save_json_data('./leaderboard_data.js', score_table)


def cal_image_counts(data):
    type_count_dic = defaultdict(list)
    for item in data:
        task_name = item['id'].split('_')[0]
        count =  len([f for f in os.listdir(item['img_path']) if f.lower().endswith('.png')])
        type_count_dic['all'].append(count)
        for type_name in TYPE_TASK_MAP.keys():
            if task_name in TYPE_TASK_MAP[type_name]:
                type_count_dic[type_name].append(count)
                break
    split_img_counts = {}
    for k, v in type_count_dic.items():
        p1, p2 = np.percentile(v, [33, 66])
        if p1 == p2:
            p2 += 0.1
        split_img_counts[k] = [p1, p2]
    print(split_img_counts)
    return split_img_counts
            
            
def recap(result):
    for item in result:
        preds = extract_answer(response_text=item['response'])
        # pred_answer = get_final_pred(preds) if preds else None
        pred_answer = extract_answer([item['best_response']])[0]
        golden_answer = item['golden_answer']
        cor_flag = [pred == golden_answer for pred in preds]
        correct = pred_answer == golden_answer
        item['preds'] = preds
        item['pred_answer'] = pred_answer
        item['cor_flag'] = cor_flag
        item['correct'] = correct
    return result 

def recap_answer(args):
    
    output_dir = f'./result/rm/'
    if args.model == 'all':
        paths = os.listdir(output_dir)
        for path in paths:
            result_path = os.path.join(output_dir, path)
            result = load_json_data(result_path)
            result = recap(result)
            save_json_data(result_path, result)
    else:
        output_path = os.path.join(output_dir, f"{args.model}_{args.method}.json")
        result = load_json_data(output_path)
        result = recap(result)
        save_json_data(output_path, result)


def split_task(args):
    full_dir = f'./result/main_exp/all/'
    full_path = os.path.join(full_dir, f"{args.model}_{args.method}.json")
    full_results = load_json_data(full_path)
    task_results = defaultdict(list)
    for item in full_results:
        if 'id' not in item:
            continue
        task = item['id'].split('_')[0]
        task_results[task].append(item)
    print(f'{args.model}')
    for task, results in task_results.items():
        corrects = sum(1 for item in results if item['cor_flag'] and random.choice(item['cor_flag']))
        maj_corrects =  sum(1 for item in results if item['correct'])
        metric ={
            'correct': corrects,
            'maj_correct': maj_corrects,
            'length': len(results),
            'acc': round(corrects / len(results), 5),
            'maj_acc': round(maj_corrects / len(results), 5)
        }
        print(f"{task}: {metric['acc']}")
        results.append(metric)
        output_dir = os.path.join('./result/main_exp/', TASK_PATH_MAP[task])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"{args.model}_{args.method}.json")
        save_json_data(output_path, results)
        
def get_type_acc(result):
    type_results = defaultdict(list)
    for item in result:
        task_name = item['id'].split('_')[0]
        for type in TYPE_TASK_MAP:
            if task_name in TYPE_TASK_MAP[type] and item['cor_flag']:
                type_results[type].append(item['cor_flag'][0]) 
                break 
    metric = {}
    for k in TYPE_TASK_MAP:
        v = type_results[k]
        acc = np.mean(np.array(v)).item()
        metric[k] = acc
    return metric


def get_model_result(full_result, args):
    metrics = {type:{'correct':0, 'maj_correct':0, 'length':0} for type in TYPE_TASK_MAP.keys()}
    metrics['all'] = {'correct':0, 'maj_correct':0, 'length':0}
    task_results = defaultdict(list)
    wrong_index  = [
        "HAL_17", "ARI_68", "MCA_147", "HAL_108", "HAL_32", 
        "MCD_71", "SFI_5", "SFI_17", "SFI_27", "SFI_39", 
        "SFI_51", "SFI_61", "SFI_85", "SFI_105", "SFI_115", 
        "SFI_116", "SFI_119", "SFI_125", "SFI_149", "SFI_172", 
        "SFI_187", "PSI_20", "PSI_24", "PSI_27", "PSI_83", 
        "PSI_84", "PSI_99", "PSI_121", "PSI_140", "PSI_161"
    ]
    for item in full_result:
        if 'id' not in item or item['id'] in wrong_index:
            continue
        task = item['id'].split('_')[0]
        task_results[task].append(item)
    for task, results in task_results.items():
        if len(results[0]['cor_flag']) > 1:
            corrects = 0
            for item in results:
                if not item['cor_flag']:
                    continue
                sample_cnt = min(5, len(item['cor_flag']))
                correct = random.sample(item['cor_flag'], sample_cnt)
                if sum(correct) > 2:
                    corrects += 1
        else:
            corrects = sum(1 for item in results if item['cor_flag'] and random.choice(item['cor_flag']))
        maj_corrects =  sum(1 for item in results if item['correct'])
        for type, metric in metrics.items():
            if task in TYPE_TASK_MAP[type]:
                metric['correct'] += corrects
                metric['maj_correct'] += maj_corrects
                metric['length'] += len(results)
                break 
        metrics['all']['correct'] += corrects
        metrics['all']['maj_correct'] += maj_corrects
        metrics['all']['length'] += len(results)
        
    print(f'{args.model}')
    result = {'Model': [args.model]}
    for type, metric in metrics.items():
        if args.method != 'sc':
            acc = round(metric['correct'] * 100 / metric['length'], 2)
        else:
            acc = round(metric['maj_correct'] * 100 / metric['length'], 2)
        print(f'{type}: {acc}')
        result[f'{type}'] = [acc]
    return result
    
    
def get_main_result(args):
    method = 'direct' if args.method == 'direct' else 'cot'

    if args.model == 'all':
        all_metrics = defaultdict(list)
        paths = os.listdir('./result/main_exp/all/')
        for path in paths:
            if method not in path:
                continue
            str_ls = os.path.splitext(path)[0].split('_')[:-1]
            args.model = ('_').join(str_ls)
            url = MODEL_URL[args.model] if args.model in MODEL_URL.keys() else None
            result_path = os.path.join('./result/main_exp/all/', path)
            full_result = load_json_data(result_path)

            metric = get_model_result(full_result, args)
            all_metrics['url'] += [url]
            for k, v in metric.items():
                if k == 'Model':
                    v = transform_model_name(v)
                all_metrics[k] += v
        if args.leaderboard:
            get_leaderboard_results(all_metrics)
        all_metrics = pd.DataFrame(all_metrics)
        all_metrics.to_csv("all_result.csv", index=False, encoding="utf-8-sig")
    else:    
        full_result_path = f'./result/main_exp/all/{args.model}_{method}.json'
        full_result = load_json_data(full_result_path)
        metric = get_model_result(full_result, args)
        metric = pd.DataFrame(metric)
        metric.to_csv(f"{args.model}_result.csv", index=False, encoding="utf-8-sig")

def transform_model_name(model):
    if model == 'claude-3.7-sonnet':
        return 'Claude-3.7-Sonnet'
    elif model == 'claude-sonnet-4':
        return 'Claude-Sonnet-4'
    elif model == 'claude-sonnet-4-thinking':
        return 'Claude-Sonnet-4-Thinking'
    elif model == 'gemini-2.5-pro':
        return 'Gemini-2.5-Pro',
    elif model == 'claude-sonnet-4':
        return 'llava-1.5-7b'
    elif model == 'Llava-1.5-7B':
        return 'Gemini-2.5-Flash'
    elif model == 'doubao-1.5-vision':
        model = 'Doubao-1.5-Vision'
    elif model == 'gpt-4.1':
        return 'GPT-4.1'
    elif model == 'gpt-4o':
        return 'GPT-4o'
    elif model == 'gpt-5':
        return 'GPT-5'
    elif model == 'gpt-5-mini':
        return 'GPT-5-mini'
    elif model == 'InternVL3_5-8B':
        return 'InternVL3.5-8B'
    elif model == 'InternVL3_5-30B-A3B':
        return 'InternVL3.5-30B-A3B'
    elif model == 'InternVL3_5-38B':
        return 'InternVL3.5-38B'
    else:
        return model

def comp_token_acc(args):
    SELECT_MODELS = ['gpt-5-mini', 'o4-mini', 'gemini-2.5-flash', 'gemini-2.5-pro',
                     'gpt-5', 'claude-3.7-sonnet','gpt-4.1', 'VL-Rethinker-72B', 
                    'MiMo-VL-7B-RL', 'Gemma3-27B','QVQ-72B-Preview',
                    'Qwen2.5-VL-7B', 'MM-Eureka-Qwen-32B', 'Keye-VL-1.5-8B']
    
    result_dir = './result/main_exp/all/'
    metrics = {}
    if args.type == 'all':
        tasks = [x for v in TYPE_TASK_MAP.values() for x in v]
    else:
        tasks = TYPE_TASK_MAP[args.type]
    encoding = tiktoken.encoding_for_model("gpt-4")
    for model in SELECT_MODELS:
        path = os.path.join(result_dir, f"{model}_cot.json")
        result = load_json_data(path)
        tokens = [len(encoding.encode(item['response'][0])) for item in result if item['id'].split('_')[0] in tasks]
        correct = [item['cor_flag'][0] for item in result if item['cor_flag'] and item['id'].split('_')[0] in tasks]
        tokens = np.mean(np.array(tokens)).item()
        if model in ['gemini-2.5-flash', 'gpt-5-mini', 'o4-mini', 'gpt-5', 'gemini-2.5-pro']:
            path = os.path.join('./result/usage/', f"{model}_cot.json")
            result = load_json_data(path)
            tokens = [item['completion_tokens'] for item in result if item['id'].split('_')[0] in tasks]
            tokens = np.mean(np.array(tokens)).item()
        metrics[tokens] = {}
        acc = np.mean(np.array(correct)).item()
        metrics[tokens][transform_model_name(model)] = acc
    data = prepare_fig_input(metrics, ['Tokens', 'Acc', 'Model'])
    print(data)
    draw_scatter(data=data, path=f'./figures/{args.type}_token_acc_comp.pdf')
    
    
def comp_type_token():
    result_dir = './result/main_exp/all/'
    SELECT_MODELS = ['gpt-5', 'gemini-2.5-pro', 'gemini-2.5-flash',
                      'o4-mini', 'gpt-5-mini']
    
    result_dir = './result/main_exp/all/'
    metrics = defaultdict(dict)
    encoding = tiktoken.encoding_for_model("gpt-4")
    for model in SELECT_MODELS:
        type_tokens = defaultdict(list)
        if model in ['gemini-2.5-flash', 'gpt-5-mini', 'o4-mini', 'gpt-5', 'gemini-2.5-pro']:
            path = os.path.join('./result/usage/', f"{model}_cot.json")
            result = load_json_data(path)
            for item in result:
                for k, v in TYPE_TASK_MAP.items():
                    if item['id'].split('_')[0] in v:
                        type_tokens[k].append(item['completion_tokens'])
                        break
        else:
            path = os.path.join(result_dir, f"{model}_cot.json")
            result = load_json_data(path)
            for item in result:
                for k, v in TYPE_TASK_MAP.items():
                    if item['id'].split('_')[0] in v:
                        type_tokens[k].append(len(encoding.encode(item['response'][0])))
                        break
            
        for k in TYPE_TASK_MAP:
            tokens = np.mean(np.array(type_tokens[k])).item()
            metrics[k][transform_model_name(model)] = tokens
    data = prepare_fig_input(metrics, ['Type', 'Tokens', 'Model'])
    print(data)
    draw_line(data=data, path='./figures/type_token_comp.pdf')



def comp_nothinking_acc():
    full_dir = f'./result/main_exp/all/'
    NO_THINKING_MODELS = ['Qwen2.5-VL-72B', 'Gemma3-27B', 'gpt-4.1', 'Qwen2.5-VL-32B', 'gpt-4o']
    for model in NO_THINKING_MODELS:
        metrics = {}
        direct_path = os.path.join(full_dir, f'{model}_direct.json')
        direct_data = load_json_data(direct_path)
        direct_metric = get_type_acc(direct_data)
        for k, v in direct_metric.items():
            type = k[:3].capitalize()
            metrics[type] = {'w/o CoT': v}
        cot_path = os.path.join(full_dir, f'{model}_cot.json')
        cot_data = load_json_data(cot_path)
        cot_metric = get_type_acc(cot_data)
        for k, v in cot_metric.items():
            type = k[:3].capitalize()
            metrics[type]['w/ CoT'] = v
        result = prepare_fig_input(metrics, ['Type', 'Acc', 'Thinking'])
        print(result)
        draw_bar(data=result, path=f'./figures/nothinking_acc_{model}.pdf')      
        
def comp_thinking_acc():
    full_dir = f'./result/budget/'
    THINKING_MODELS = ['gpt-5-mini', 'gemini-2.5-flash', 'gpt-5']
    for model in THINKING_MODELS:
        metrics = defaultdict(dict)
        for effort in ['minimal', 'medium', 'high']:
            path = os.path.join(full_dir, f'{model}-{effort}.json')
            data = load_json_data(path)
            metric = get_type_acc(data)
            for k, v in metric.items():
                metrics[k[:3].capitalize()][effort] =  v
        result = prepare_fig_input(metrics, ['Type', 'Acc', 'Budget'])
        print(result)
        draw_bar(data=result, path=f'./figures/thinking_acc_{model}.pdf')            
        

def cal_type_corr():
    result_dir = './result/main_exp/all/'
    metrics = defaultdict(list)
    for file in os.listdir(result_dir):
        if 'cot' not in file:
            continue
        path = os.path.join(result_dir, file)
        result = load_json_data(path)
        if len(result) < 2676:
            continue
        type_acc = get_type_acc(result)
        for k in TYPE_TASK_MAP:
            type = k[:3].capitalize()
            metrics[type].append(type_acc[k])
    data = prepare_fig_input(metrics, ['Type', 'Acc'])
    print(data)
    draw_heat(data=data, path='./figures/type_corr.pdf')
    draw_dendrogram(data=data, path='./figures/type_cluster.pdf')


def cal_method_acc(args):
    all_metrics = defaultdict(list)
    cot_path = f'./result/main_exp/all/{args.model}_cot.json'
    rm_path = f'./result/rm/{args.model}_skywork.json'
    cot_result = load_json_data(cot_path)
    rm_result = load_json_data(rm_path)
    args.model = 'CoT'
    metric = get_model_result(cot_result, args)
    for k, v in metric.items():
        all_metrics[k] += v
    args.method = 'sc'
    args.model = 'SC@8'
    metric = get_model_result(cot_result, args)
    for k, v in metric.items():
        all_metrics[k] += v
    args.model = 'BoN@8'
    metric = get_model_result(rm_result, args)
    for k, v in metric.items():
        all_metrics[k] += v
    all_metrics = pd.DataFrame(all_metrics)
    all_metrics.to_csv("method_result.csv", index=False, encoding="utf-8-sig")
    
def comp_rl_bon(args):
    SELECT_MODELS = ['MiMo-VL-7B-SFT', 'R1-Onevision-7B', 'Qwen2.5-VL-7B']
    RL_MAP = {
        'MiMo-VL-7B-SFT': 'MiMo-VL-7B-RL',
        'OpenVLThinker-7B-v1.2-sft': 'OpenVLThinker-7B-v1.2',
        'R1-Onevision-7B': 'R1-Onevision-7B-RL',
        'Qwen2.5-VL-7B': 'VL-Rethinker-7B',
        'Qwen2.5-VL-32B': 'MM-Eureka-Qwen-32B'
    }
    NAME_MAP = {
        'MiMo-VL-7B-SFT': 'MiMo-VL',
        'OpenVLThinker-7B-v1.2-sft': 'OpenVLThinker',
        'R1-Onevision-7B': 'R1-Onevision',
        'Qwen2.5-VL-7B': 'VL-Rethinker',
        'Qwen2.5-VL-32B': 'MM-Eureka'
    }
    all_metrics = defaultdict(dict)
    for model in SELECT_MODELS:
        rl_metrics = defaultdict(list)
        cot_metrics = defaultdict(list)
        bon_metrics = defaultdict(list)
        cot_path = f'./result/main_exp/all/{model}_cot.json'
        rm_path = f'./result/rm/{model}_skywork.json'
        rl_path = f'./result/main_exp/all/{RL_MAP[model]}_cot.json'
        cot_result = load_json_data(cot_path)
        rm_result = load_json_data(rm_path)
        rl_result = load_json_data(rl_path)
        # args.model = 'Base'
        metric = get_model_result(cot_result, args)
        for k, v in metric.items():
            cot_metrics[k] = v[0]
        args.model = 'RL'
        metric = get_model_result(rl_result, args)
        for k, v in metric.items():
            rl_metrics[k] = v[0]
        args.method = 'sc'
        args.model = 'BoN'
        metric = get_model_result(rm_result, args)
        for k, v in metric.items():
            bon_metrics[k] = v[0]
            
        all_metrics['Base'][f"{NAME_MAP[model]}"] = cot_metrics['all']
        all_metrics['RL'][f"{NAME_MAP[model]}"] = rl_metrics['all']
        all_metrics['BoN'][f"{NAME_MAP[model]}"] = bon_metrics['all']
        
    data = prepare_fig_input(all_metrics, ['Method', 'Acc', 'Model'])
    print(data)
    draw_line(data=data, path=f'./figures/rl_bon_comp.pdf')

def split_image(args):
    full_dir = f'./result/main_exp/all/'
    full_path = os.path.join(full_dir, f"{args.model}_{args.method}.json")
    full_results = load_json_data(full_path)
    metrics = defaultdict(lambda: defaultdict(list))
    for item in full_results:
        correct = random.choice(item['cor_flag']) if item['cor_flag'] else False
        count =  len([f for f in os.listdir(item['img_path']) if f.lower().endswith('.png')])
        p1, p2 = TYPE_SPLIT_DICT['all']
        if count <= p1:
            metrics['all']['low'].append(correct)
        elif count <= p2:
            metrics['all']['mid'].append(correct)
        else:
            metrics['all']['high'].append(correct)
        task_name = item['id'].split('_')[0]
        for type_name in TYPE_TASK_MAP.keys():
            if task_name in TYPE_TASK_MAP[type_name]:
                p1, p2 = TYPE_SPLIT_DICT[type_name]
                if count <= p1:
                    metrics[type_name]['low'].append(correct)
                elif count <= p2:
                    metrics[type_name]['mid'].append(correct)
                else:
                    metrics[type_name]['high'].append(correct)
                break 
            
    print(f'{args.model}')
    result = {}
    fig_path = f'./figures/{args.model}_image_count.pdf'
    for type in TYPE_TASK_MAP.keys():
        metric = metrics[type]
        type = type[:3].capitalize()
        result[type] = {}
        for level in ['low', 'mid', 'high']:
            corrects = metric[level]
            acc =  round(sum(corrects) / len(corrects), 2)
            print(f'{level}: {acc}') 
            result[type][level] = acc
    
    data = prepare_fig_input(result, ['Type', 'Acc', 'Level'])
    print(data)
    draw_bar(data=data, path=fig_path)
   

def comp_type_recall(args):
    path = f'./result/n_scale/{args.model}_128.json'
    data = load_json_data(path)
    all_metric = {}
    for i in [2**j for j in range(8)]:
        all_metric[i] = {}
        metric = defaultdict(list)
        for item in data:
            cor = any(item['cor_flag'][:i])
            for type in TYPE_TASK_MAP:
                if item['id'].split('_')[0] in TYPE_TASK_MAP[type]:
                    metric[type].append(cor)
                    break 
        for type in TYPE_TASK_MAP:
            acc = np.mean(np.array(metric[type]))
            all_metric[i][type[:3].capitalize()] = acc
    data = prepare_fig_input(all_metric, ['k', 'Pass@k', 'Type'])
    print(data)
    draw_line(data=data, path=f'./figures/{args.model}_recall.pdf')
        
   
def main(args):
    if args.option == 'task':
        split_task(args)
    elif args.option == 'recap':
        recap_answer(args)
    elif args.option == 'image':
        split_image(args)
    elif args.option == 'token_acc':
        comp_token_acc(args)
    elif args.option == 'type_token':
        comp_type_token()
    elif args.option == 'nothinking_acc':
        comp_nothinking_acc()
    elif args.option == 'thinking_acc':
        comp_thinking_acc()
    elif args.option == 'type_corr':
        cal_type_corr()
    elif args.option == 'method':
        cal_method_acc(args)
    elif args.option == 'rl_bon':
        comp_rl_bon(args)
    else:
        get_main_result(args)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='all')
parser.add_argument('--method', type=str, default='cot')
parser.add_argument('--option', type=str, default='result')
parser.add_argument('--type', type=str, default='all')
parser.add_argument('--leaderboard', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    main(args)