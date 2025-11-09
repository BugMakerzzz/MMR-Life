#!/usr/bin/env python3
import os
from utils.load_data import load_raw_dataset, save_json_data, load_last_index, load_json_data
from tqdm import tqdm
import shutil

TASK_MAP_DICT = {
    'abductive': {
        'TVbench': 'Human Activity Attribution',
        'cartoon': 'Character Interaction Attribution',
        'cleverer': 'Multi-Hop Collision Attribution'
    },
    'analogical': {
        'animal': 'Animal Relation Inference',
        'sim_shoes': 'Product Similarity Inference',
        'dif_shoes': 'Product Similarity Inference',
        'art': 'Artwork Style Inference'
    },
    'causal': {
        'cartoon': 'Character Interaction Prediction',
        'cleverer': 'Multi-Hop Collision Prediction',
        'fluid': 'Counterfactual Fluid Prediction'
    },
    'deductive': {
        'material': 'Material Composition Deduction',
        'poke': 'Card Winner Deduction',
        'recipe': 'Recipe Step Deduction'
    },
    'inductive': {
        'bird': 'Bird Migration Induction',
        'plant': 'Plant Disease Induction',
        'sport': 'Sport Feature Induction'
    },
    'spatial': {
        'direction': 'Relative Position Estimation',
        'navi': 'Camera Rotation Estimation',
        'path': 'Navigation Route Planning' 
    },
    'temporal': {
        'crowd': 'Crowd Timeline Reconstruction',
        'driving': 'Driving Sequence Prediction',
        'TVbench': 'Human Activity Localization'
    }
}

def get_task_name(type, task):
    full_name = TASK_MAP_DICT[type][task]
    words = full_name.split(' ')
    abbreviation = ''.join([word[0].upper() for word in words])
    return abbreviation, full_name

def move_img_files(src_dir, tgt_dir):
    os.makedirs(tgt_dir, exist_ok=True)
    for file in os.listdir(src_dir):
        if file.lower().endswith(".png"):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(tgt_dir, file)
            shutil.copy2(src_file, dst_file) 
            
    if os.path.exists(os.path.join(src_dir, 'options')):
        src_option = os.path.join(src_dir, 'options')
        tgt_option = os.path.join(tgt_dir, 'options')
        os.makedirs(tgt_option, exist_ok=True)
        for file in os.listdir(src_option):
            if file.lower().endswith(".png"):
                src_file = os.path.join(src_option, file)
                dst_file = os.path.join(tgt_option, file)
                shutil.copy2(src_file, dst_file) 

def select_easy_questions(type, task):
    def load_items(path):
        data = load_json_data(path)[:-1]
        items = []
        for x in data:
            _id = x.get("index")
            corr = x.get("correct")
            items.append({"id": _id, "correct": corr})
        return items
    
    files = [f'./raw_result/{type}/{task}/Gemma3-4B_cot.json', 
             f'./raw_result/{type}/{task}/InternVL3_5-8B_cot.json', 
             f'./raw_result/{type}/{task}/Qwen2.5-VL-7B_cot.json']
    data = [load_items(f) for f in files]
    per_file_maps = []
    for items in data:
        m = {}
        for x in items:
            m[x["id"]] =  x
        per_file_maps.append(m)
    ids_all = set(per_file_maps[0]) & set(per_file_maps[1]) & set(per_file_maps[2])
    report = []
    for _id in sorted(ids_all):
        rows = [m[_id] for m in per_file_maps]
        all_correct = all(r["correct"] for r in rows)
        if not all_correct:
            continue
        print(_id)
        report.append(_id)
    return report

def main():
    root_dir = './raw_question/'
    data = []
    merge_cnt = 0
    for type in os.listdir(root_dir):
        for task in os.listdir(os.path.join(root_dir, type)):
            if task != 'fluid':
                continue
            # easy_index = select_easy_questions(type, task)
            input_dir = os.path.join(root_dir, type, task)
            dataset = load_raw_dataset(data_path=input_dir, 
                                system_prompt='', 
                                n_examples=10000)
            final_idx = merge_cnt if merge_cnt and task in ['sim_shoes', 'dif_shoes'] else 1
            for i, item in tqdm(enumerate(dataset)):
                idx = f"Q_{type.upper()}_{task.upper()}_{i+1}"
                # if idx in easy_index:
                #     continue
                question = item['question']
                golden_answer = item['golden_answer']
                source = item['source']
                task_name, full_task_name = get_task_name(type, task)
                if task_name != 'CFP':
                    continue
                src_dir = os.path.join(input_dir, str(source))
                tgt_dir = f"./image/{type}/{full_task_name}/{final_idx}"
                move_img_files(src_dir=src_dir, tgt_dir=tgt_dir)
                msg = {
                    'id': f"{task_name}_{final_idx}",
                    'img_path': tgt_dir,
                    'question': question,
                    'golden_answer': golden_answer
                }
                data.append(msg)
                final_idx += 1
            if task in ['dif_shoes', 'sim_shoes']:
                merge_cnt = final_idx
    
    data_path = './DL-MMR_CFP.json'
    save_json_data(data_path, data)
    print(f"Dataset saved to {data_path}")


if __name__ == "__main__":
    main()

