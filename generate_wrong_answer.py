from utils.load_data import load_json_data, save_json_data
import os 
import re
import argparse
from sentence_transformers import SentenceTransformer, util
import random 
from collections import defaultdict
random.seed(17)

def process_order_answers(wrong_dic, source, golden_answer):
    
    def complete_string(string, n):
        numbers = string.split('-')
        current_length = len(numbers)
        if current_length < n:
            # 生成一个目标长度为 n 的数字集合
            all_numbers = list(range(1, n + 1))
            all_numbers = set([str(k) for k in all_numbers])
            missing_numbers = all_numbers - set(numbers)  # 找出缺失的数字
            # 补充缺失的数字到字符串末尾
            for num in sorted(missing_numbers):
                string += f'-{num}'
        
        return string
    
    answers = wrong_dic[source]
    if not answers:
        return [], []
    wrong_answer = set()
    for ans in answers:
        ans = complete_string(ans, len(golden_answer.split('-')))
        wrong_answer.add(ans)
    wrong_answer = list(wrong_answer)
    wrong_source = len(wrong_answer) * [source]             
    return wrong_answer, wrong_source


def process_degree_answers(wrong_dic, source):
    standard_degree = ['15', '30', '45', '60', '90', '105', '120', '135', '150', '180']
    def filter_degrees(string):
        str_ls = string.split(',')
        ans_ls = []
        for str in str_ls:
            degree_matches = re.findall(r'(\d+)\s+degree', str)
            if degree_matches:
                degree = degree_matches[0]
                if degree == '0':
                    continue
                elif degree == '180':
                    str.replace('counterclockwise', 'clockwise')
                elif degree not in standard_degree:
                    return None 
            ans_ls.append(str)
        return (',').join(ans_ls)    
    
    answers = wrong_dic[source]
    if not answers:
        return [], []
    wrong_answer = set()
    for ans in answers:
        ans = filter_degrees(ans)
        if ans:
            wrong_answer.add(ans)
    wrong_answer = list(wrong_answer)
    wrong_source = len(wrong_answer) * [source]             
    return wrong_answer, wrong_source


def filter_wrong_answers(args):
    root_dir = f'./raw_question/{args.type}/{args.task}/'
    data = load_json_data(os.path.join(root_dir, 'question_with_wrong.json'))
    results = []
    for item in data:
        wrong_dic = defaultdict(list)
        for i in range(len(item['wrong_answer'])):
            answer = item['wrong_answer'][i]
            wrong_dic[item['wrong_source'][i]].append(answer)
        if args.task in ['recipe', 'crowd']:
            ans1, src1 = process_order_answers(wrong_dic=wrong_dic, source='gpt-5-mini', golden_answer=item['golden_answer'])
            ans2, src2 = process_order_answers(wrong_dic=wrong_dic, source='gpt-4o', golden_answer=item['golden_answer'])
            ans3, src3 = process_order_answers(wrong_dic=wrong_dic, source='Qwen2.5-VL-32B', golden_answer=item['golden_answer'])
        elif args.task == 'navi':
            ans1, src1 = process_degree_answers(wrong_dic=wrong_dic, source='gpt-5-mini')
            ans2, src2 = process_degree_answers(wrong_dic=wrong_dic, source='gpt-4o')
            ans3, src3 = process_degree_answers(wrong_dic=wrong_dic, source='Qwen2.5-VL-32B')
        else:
            ans1, ans2, ans3 = wrong_dic['gpt-5-mini'], wrong_dic['gpt-4o'], wrong_dic['Qwen2.5-VL-32B']
            src1, src2, src3 = len(ans1) * ['gpt-5-mini'], len(ans2) * ['gpt-4o'], len(ans3) * ['Qwen2.5-VL-32B']
            direction = ['North', 'South', 'East', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest']
            random.shuffle(direction)
            ans3 += direction
            src3 += ['random'] * 7
        ans = ans1 + ans2 + ans3 
        src = src1 + src2 + src3
        wrong_answers = []
        wrong_sources = []
        for i in range(len(ans)):
            if ans[i] == item['golden_answer'] or ans[i] in wrong_answers:
                continue
            wrong_answers.append(ans[i])
            wrong_sources.append(src[i])
        if len(wrong_answers) < 4:
            continue
        item['wrong_answer'] = wrong_answers[:4]
        item['wrong_source'] = wrong_sources[:4]
        results.append(item)
    save_json_data(os.path.join(root_dir, 'question_with_wrong_final.json'), results)
        


def extract_answer(response_text):
    """Extract answers from the response"""
    if not response_text:
        return []
    answers = []
    for text in response_text:
        text = text.strip()
        text = re.sub(r'[*]\s*', '', text)  # Remove "- ", "* ", etc.
        # Remove list markers like "1.", "2.", "-", "*", etc.
        match = re.search(r'Answer:\s*(.*)', text)
        if match:
            cleaned_text = match.group(1)
        else:
            continue
        if cleaned_text:
            answers.append(cleaned_text)
    return answers

def extract_unique_answers(task, model, sentences, golden_answer):
    if task in ['direction', 'navi', 'path', 'recipe', 'crowd']:
        sentences = [sent.strip('.') for sent in sentences]
        sentences = [sent.strip('-') for sent in sentences]
        sentences = [sent.strip() for sent in sentences]
        if task in ['recipe', 'crowd']:
            length = len(golden_answer.split('-'))
            answers = sentences.copy()
            sentences = []
            for answer in answers:
                if not all(ch.isdigit() or ch == "-" for ch in answer):
                    continue
                if len(answer.split('-')) > length:
                    answer_ls = answer.split('-')
                    if str(length+1) in answer_ls:
                        answer_ls.remove(str(length+1))
                    elif '0' in answer_ls:
                        answer_ls.remove('0')
                    answer = ('-').join(answer_ls)
                sentences.append(answer)
        elif task in ['direction']:
            pattern = re.compile("(northeast|northwest|southeast|southwest|north|south|east|west)", re.IGNORECASE)
            answers = []
            for s in sentences:
                match = pattern.search(s)   # search 只找第一个匹配
                if match:
                    answer = match.group()
                    answers.append(answer.lower().capitalize())
            sentences = answers
        unique_sentences = list(set(sentences))
    else:
        embeddings = model.encode(sentences, convert_to_tensor=True)

        # 计算相似度矩阵
        cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

        # 根据相似度阈值去重
        threshold = 0.9
        unique_sentences = []
        seen = set()

        for i, s in enumerate(sentences):
            if any(cosine_scores[i][j] > threshold for j in seen):
                continue
            unique_sentences.append(s)
            seen.add(i)

    return unique_sentences

def get_wrong_responses(args, base_dir, model):
    raw_question_path = os.path.join(base_dir, 'question.json')
    question_data = load_json_data(raw_question_path)
    for filename in os.listdir(base_dir):
        if filename.startswith('neg_option') and filename.endswith('.json'):
            pattern = re.compile(r"option_(.*?)\.json")
            source = pattern.search(filename).group(1) 
            neg_option_path = os.path.join(base_dir, filename)
            neg_option_data = load_json_data(neg_option_path)
            
            for neg_item in neg_option_data:
                index = neg_option_data.index(neg_item)  # 假设每个项都有 "index"
                question_item = question_data[index]
                if 'wrong_answer' not in question_item:
                    question_item['wrong_answer'] = []
                    question_item['wrong_source'] = []
                wrong_answers = extract_answer(neg_item['wrong_response'])
                wrong_answers = extract_unique_answers(args.task, model, wrong_answers, question_item['golden_answer'])
                question_item['wrong_answer'] += wrong_answers
                question_item['wrong_source'] += [source] * len(wrong_answers)
    output_path = os.path.join(base_dir, 'question_with_wrong.json')
    save_json_data(output_path, question_data)


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='spatial')
parser.add_argument('--type', type=str, default='causal')
parser.add_argument('--final', action='store_true')
args = parser.parse_args() 


if __name__ == '__main__':
    if args.final:
        filter_wrong_answers(args)
    else:
        base_dir = f'./raw_question/{args.type}/{args.task}'
        model = SentenceTransformer('/netcache/huggingface/all-MiniLM-L6-v2')
        get_wrong_responses(args,base_dir, model)