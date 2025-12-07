from sentence_transformers import SentenceTransformer, util
from PIL import Image
from utils.load_data import load_json_data, get_example_images, save_json_data
from tqdm import tqdm 
import os 
import re
import torch 
from transformers import CLIPProcessor, CLIPModel

def load_dataset():
    dataset = load_json_data('./MMR_Life.json')
    data = []
    for i, example in tqdm(enumerate(dataset), total=len(dataset), desc="Loading dataset"):
        image_paths = get_example_images(example['img_path'])
        option_path = os.path.join(example['img_path'], 'options') 
        option_image_paths = get_example_images(option_path) if os.path.exists(option_path) else None 
        question, option_str = example['question'].split('\nOptions: ')
        options = re.split(r"[A-E]\.\s*", option_str)
        if not options:
            options = None
        else:
            options = options[1:]
            options = [f'{question}\nAnswer: {option}' for option in options]
        msg = {
            'id':example['id'],
            'question':question,
            'text_options': options,
            'image': image_paths,
            'image_options': option_image_paths,
            'answer': example['golden_answer']
        }
        data.append(msg)
    return data
        
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'clip-ViT-B-32'
model_path = f'/netcache/huggingface/{model_name}'
model = SentenceTransformer(model_path, device=device)
dataset = load_dataset()

results = []
for item in tqdm(dataset):
    img_embs = []
    for path in item['image']:
        emb = model.encode(
            Image.open(path),
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        )
        img_embs.append(emb)
    img_embs = torch.stack(img_embs)   # shape: [N, D]
    avg_img_emb = img_embs.mean(dim=0)  # shape: [D]     
    if item['text_options']:
        opt_embs = model.encode(
            item['text_options'],
            convert_to_tensor=True,
            device=device
        )
    else:
        opt_embs = []
        for path in item['image_options']:
            emb = model.encode(
                Image.open(path),
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False
            )
            opt_embs.append(emb)
        opt_embs = torch.stack(opt_embs, dim=0)
    scores = util.cos_sim(avg_img_emb, opt_embs)  
    best_idx = torch.argmax(scores).item()
    pred = chr(ord('A') + best_idx)
    correct = pred == item['answer']
    msg = {
            'id': item['id'],
            'question': item['question'],
            'preds': [pred],
            'pred_answer': pred,
            'golden_answer': item['answer'],
            'cor_flag': [correct],
            'correct': correct
    }
    results.append(msg)

output_dir = './result/main_exp/all/'
    
output_path = os.path.join(output_dir, f"{model_name}.json")

save_json_data(output_path, results)