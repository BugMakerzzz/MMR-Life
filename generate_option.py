import os 
import random
random.seed(17)
from utils.load_data import load_json_data, save_json_data
from prompt.generate_option_prompt import user_prompt_dic
import argparse
answer_mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E', 
                1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E',
                'Throughout the entire sequence.':'A', 
                'At the end of the sequence.':'B', 
                'In the middle of the sequence.':'C', 
                'At the beginning of the sequence.':'D',
                'Not shown in the sequence.':'E', }

def generate_option_questions(args):
    question_dir = f'./raw_question/{args.type}/{args.task}/'
    data = load_json_data(os.path.join(question_dir, 'question.json'))
    wrong_dic = {}
    if args.task in ['cartoon', 'TVbench', 'crowd', 'recipe', 'direction', 'navi', 'path']:
        wrong_answers = load_json_data(os.path.join(question_dir, 'question_with_wrong.json'))
        if wrong_answers:
            for item in wrong_answers:
                options = item['wrong_answer'].copy()
                random.shuffle(options)
                correct_idx = random.randint(0, 4)
                options.insert(correct_idx, item['golden_answer'])  
                golden_answer = answer_mapping[correct_idx + 1]
                wrong_dic[f"{item['source']}"] = {'option':options, 'answer':golden_answer}
    results = []
    for item in data:
        golden_answer = item.get('golden_answer')
        if golden_answer and golden_answer in answer_mapping:
            item['golden_answer'] = answer_mapping[golden_answer]
        if args.task == 'material':
            item['options'] = ['1', '2', '3', '4', 'None of the above']
        elif args.type == 'temporal' and args.task == 'TVbench':
            item['options'] = ['Throughout the entire sequence.', 'At the end of the sequence.', 'In the middle of the sequence.', 'At the beginning of the sequence.', 'Not shown in the sequence.']
        elif args.task in ['cartoon', 'TVbench', 'crowd', 'recipe', 'direction', 'navi', 'path'] and wrong_dic:
            key = f"{item['source']}"
            if key not in wrong_dic.keys():
                continue
            item['options'] = wrong_dic[key]['option']
            item['golden_answer'] = wrong_dic[key]['answer']
        if user_prompt_dic[args.type][args.task]:
            item['question'] = random.choice(user_prompt_dic[args.type][args.task])
        results.append(item)
    new_file_path = os.path.join(question_dir, 'question_with_option.json')
    save_json_data(new_file_path, results)

    print(f"Processed:{new_file_path}")

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='abductive')
parser.add_argument('--task', type=str, default='cartoon')
args = parser.parse_args()
if __name__ == '__main__':
    generate_option_questions(args)
  