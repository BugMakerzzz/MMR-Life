import json 
import os
from .config import *
import base64
import glob
import natsort 
import random 
from tqdm import tqdm
import re
from collections import Counter
random.seed(17)


def encode_image(image_path):
    """Encode a single image file"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def get_example_images(image_folder_path, masked=False):

    if not os.path.exists(image_folder_path):
        print(f"❌ Folder {image_folder_path} does not exist")
        return []

    # Get all image files
    image_files = glob.glob(os.path.join(image_folder_path, "*.png"))
    image_files = natsort.natsorted(image_files)  # Ensure the order is consistent
    
    if masked:
        num_to_mask = len(image_files) // 2
        removed_elements = random.sample(image_files, num_to_mask)
        for element in removed_elements:
            image_files.remove(element)
            
    return image_files

def load_oe_dataset(data_path, system_prompt, n_examples, masked=False):
    dataset = load_json_data(os.path.join(data_path, 'question.json'))[:n_examples]
    data = []
    for i, example in enumerate(dataset):
        image_paths = get_example_images(os.path.join(data_path, str(example['source'])), masked=masked)
        if not image_paths:
            print(f"❌ No image files found for example {i+1}")
            return None
        
        question = example['question']
    
        # English prompt creation
        input = f'Question: {question}' 
        content = [{"type": "text", "text": input}]

        for image_path in image_paths:
            try:
                base64_image = encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            except Exception as e:
                print(f"❌ Image loading fail {image_path}: {e}")
                continue
        
        message=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]
        example['input'] = message
        # Add a small delay between API calls to avoid rate limits
        data.append(example)
    return data 

def embed_openai_input(example, image_paths, option_image_paths, system_prompt):
    content = []
    for image_path in image_paths:
        try:
            base64_image = encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })
        except Exception as e:
            print(f"❌ Image loading fail {image_path}: {e}")
            continue
    
    input = example['question']
    content.append({"type": "text", "text": input})
    if option_image_paths:
        
        for image_path in option_image_paths:
            try:
                base64_image = encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
                content.append({
                    "type": "text",
                    "text": str(chr(65 + option_image_paths.index(image_path)))
                })
            except Exception as e:
                print(f"❌ Image loading fail {image_path}: {e}")
                continue
    message=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]
    return message


def load_raw_dataset(data_path, system_prompt, n_examples=500):
    dataset = load_json_data(os.path.join(data_path, 'question_with_option.json'))[:n_examples]
    data = []
    for i, example in enumerate(dataset):
        image_paths = get_example_images(os.path.join(data_path, str(example['source'])))
        option_path = os.path.join(data_path, str(example['source']), 'options') 
        option_image_paths = get_example_images(option_path) if os.path.exists(option_path) else None 
        if not image_paths:
            print(f"❌ No image files found for example {i+1}")
            return None
        example['question'] = 'Question: ' + example['question'] + '\nOptions: '
        if 'options' in example:
            for i, content in enumerate(example['options']):
                example['question'] += f'{str(chr(65 + i))}. {content} '
        
        message = embed_openai_input(example=example, 
                                    image_paths=image_paths, 
                                    option_image_paths=option_image_paths,
                                    system_prompt=system_prompt)
        example['input'] = message
        # Add a small delay between API calls to avoid rate limits
        data.append(example)
    return data 


def load_dataset(system_prompt, n_examples=3000, shuffle=False, mini=False):
    if mini:
        dataset = load_json_data('./MMR_Life_mini.json')
    else:
        dataset = load_json_data('./MMR_Life.json')
    if shuffle:
        random.shuffle(dataset)
    dataset = dataset[:n_examples]
    data = []
    for i, example in tqdm(enumerate(dataset), total=len(dataset), desc="Loading dataset"):
        image_paths = get_example_images(example['img_path'])
        # print(image_paths)
        option_path = os.path.join(example['img_path'], 'options') 
        option_image_paths = get_example_images(option_path) if os.path.exists(option_path) else None 
        if not image_paths:
            print(f"❌ No image files found for example {i+1}")
            return None
        message = embed_openai_input(example=example, 
                                            image_paths=image_paths, 
                                            option_image_paths=option_image_paths,
                                            system_prompt=system_prompt)
        example['input'] = message
        # Add a small delay between API calls to avoid rate limits
        data.append(example)
    return data 

def load_rm_dataset(n_examples=3000, shuffle=False):
    dataset = load_json_data('./DL-MMR.json')[:n_examples]
    if shuffle:
        random.shuffle(dataset)
    data = []
    for i, example in tqdm(enumerate(dataset), total=len(dataset), desc="Loading dataset"):
        image_files = glob.glob(os.path.join(example['img_path'], "*.png"))
        image_files = natsort.natsorted(image_files)
        option_path = os.path.join(example['img_path'], 'options') 

        content = []
        for image_path in image_files:
            try:
                content.append({
                    "type": "image",
                    "image": image_path
                })
            except Exception as e:
                print(f"❌ Image loading fail {image_path}: {e}")
                continue
    
        input = example['question']
        content.append({"type": "text", "text": input})
        if os.path.exists(option_path):
            option_files = glob.glob(os.path.join(option_path, "*.png"))
            option_files = natsort.natsorted(option_files)
            for image_path in option_files:
                try:
                    content.append({
                        "type": "image",
                        "image": image_path
                    })
                    content.append({
                        "type": "text",
                        "text": str(chr(65 + option_files.index(image_path)))
                    })
                except Exception as e:
                    print(f"❌ Image loading fail {image_path}: {e}")
                    continue
        example['input'] = content
        # Add a small delay between API calls to avoid rate limits
        data.append(example)
    return data 

def load_json_data(path):
    if not os.path.exists(path):
        print(f'path {path} not exists')
        return None
    with open(path, 'r') as f:
        data = json.load(f)
        f.close()
    return data

def save_json_data(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        f.close()
    return 

def load_last_index(filepath):
    if not os.path.exists(filepath):
        return [], 0 
    with open(filepath, "r") as f:
        data = json.load(f)
    return data, len(data) 

def load_remain_data(filepath, dataset):
    if not os.path.exists(filepath):
        return [], dataset
    with open(filepath, "r") as f:
        result = json.load(f)
    ids = []
    new_result = []
    for item in result:
        # if item ['preds'] == []:
        if item ['response'] == [""]:
            continue
        ids.append(item['id'])
        new_result.append(item)
    remain_data = []
    for item in dataset:
        if item['id'] in ids:
            continue
        remain_data.append(item)
    return new_result, remain_data

def extract_answer(response_text):
    """Extract answers from the response"""
    ANSWER_REGEX = re.compile(
        r"(?:\bAnswer\s*:\s*([A-Ea-e]))|(?:\\boxed\{\s*([A-Ea-e])\s*\})|(?:([A-Ea-e])\s*\.)"
    )
    if not response_text:
        return []
    answers = []
    for text in response_text:
        text = text.strip()
        text = re.sub(r'[-*]\s*', '', text)  # Remove "- ", "* ", etc.
        last_match = None
        for m in ANSWER_REGEX.finditer(text):
            last_match = m
        if last_match:
            ch = last_match.group(1) or last_match.group(2) or last_match.group(3)
            answers.append(ch.upper())
        else:
            answers.append("")
    return answers

def get_final_pred(preds):
    counter = Counter(preds)
    pred = counter.most_common(1)[0][0]
    return pred 