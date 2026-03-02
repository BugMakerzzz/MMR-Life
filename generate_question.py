import random 
import json 
import argparse
import os 
import shutil
from collections import defaultdict
from PIL import Image
import natsort
from utils.load_data import load_json_data, save_json_data
import cv2
import numpy as np
random.seed(17)

ANIMAL_LIST = ["antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat", "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog", "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish", "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"]


def process_video(root_dir, target_dir):

    def sample_frames_from_video(video_path, num_frames):
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频的总帧数和帧率
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 计算需要采样的帧的索引
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # 设置视频的当前帧
            ret, frame = cap.read()  # 读取该帧
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames

    num_frames = 8
    for dirpath, _, filenames in os.walk(root_dir):
        new_dir = os.path.join(target_dir, os.path.basename(dirpath))
        for filename in filenames:
            if filename.lower().endswith('.mp4') and 'seg' not in filename:
                video_path = os.path.join(dirpath, filename)
                print(f"Processing video: {video_path}")
                frames = sample_frames_from_video(video_path, num_frames)
                os.makedirs(new_dir)
                for i, frame in enumerate(frames):
                    frame_filename = os.path.join(new_dir, f"{i+1}.png")
                    cv2.imwrite(frame_filename, frame)
                print(f"Extracted 8 frames from {filename}")


def copy_figure_path(path_list, target_dir):
    if not os.path.exists(target_dir):
            os.makedirs(target_dir)
    for i in range(len(path_list)):
        path = path_list[i]
        if os.path.isfile(path):
            target_path = os.path.join(target_dir, f'{i+1}.png')
            if os.path.splitext(path)[1] != '.png':
                with Image.open(path) as img:
                    img.save(target_path, 'PNG')
            else:
                shutil.copy2(path, target_path)

def get_figure_path(root_dir, dir_name, sample=1):
    if isinstance(dir_name, list):
        dir_paths = [os.path.join(root_dir, name) for name in dir_name if os.path.exists(os.path.join(root_dir, name))]
        output_files = []
        while True:
            for dir_path in dir_paths:
                files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
                file = random.choice(files)
                if file not in output_files:
                    output_files.append(file)
                if len(output_files) >= sample:
                    return output_files
    else:
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.exists(dir_path):
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            return random.sample(files, sample)

def generate_poke_questions(target_path):
    data = load_json_data('./raw_question/deductive/poke/texas_holdem_simplified_200.json')
    question_prompt = """We have 5 players in a poker game. Each player has 2 hole cards, and there are 5 community cards on the table. Determine the winner based on the best 5-card hand formed using any combination of hole cards and community cards.
    Hand rankings (from highest to lowest):
    1. Royal Flush: Five consecutive cards of the same suit, from 10 to Ace (10-J-Q-K-A).
    2. Straight Flush: Five consecutive cards of the same suit.
    3. Four of a Kind: Four cards of the same rank.
    4. Full House: Three cards of one rank and two cards of another rank.
    5. Flush: Five cards of the same suit that are not in sequential order.
    6. Straight: Five consecutive cards of different suits.
    7. Three of a Kind: Three cards of the same rank.
    8. Two Pair: Two cards of one rank and two cards of another rank.
    9. One Pair: Two cards of the same rank.
    10. High Card: A hand that doesn't fit into any of the above categories.
    The input images are in the following order: the community cards, Player 1's hole cards, Player 2's hole cards, Player 3's hole cards, Player 4's hole cards, and Player 5's hole cards.
    Please evaluate each player's hand strength and determine the winner.
    """
    result = []
    for item in data:
        msg = {
            'source': item['id'].split('_')[-1],
            'question': question_prompt,
            'golden_answer': item['correct_answer'],
            'options': [
                'Player 1',
                'Player 2',
                'Player 3',
                'Player 4',
                'Player 5'
            ]
        }
        result.append(msg)
        
    save_json_data(target_path, result)

def generate_fluid_questions(target_path):
    data1 = load_json_data('./raw_data/fluid/question/fluid_final_test_v1.json')
    data2 = load_json_data('./raw_data/fluid/question/fluid_final_val_v1.json')
    data3 = load_json_data('./raw_data/fluid/question/fluid_final_train_v1.json')
    data = data3 + data2 + data1
    result = []
    for item in data:
        if "counterfactual" in item["question_family"]:
            golden_answers = item['positive']
            wrong_answers = item['negative']
            if not wrong_answers:
                continue
            if 'light blue' in item['question']:
                continue
            golden_count = len(golden_answers)
            remove_stick = item['program']['question'][0]
            stick = remove_stick.split()[0].capitalize() + " " + remove_stick.split()[1]
            if stick in wrong_answers:
                wrong_answers.remove(stick)
            if golden_count == 1:
                if len(wrong_answers) != 4:
                    continue
                options = wrong_answers.copy()
                correct_idx = random.randint(0, 4)
                options.insert(correct_idx, golden_answers[0])
            else:
                options = []
                for wrong in wrong_answers:
                    correct = random.sample(golden_answers, golden_count-1)
                    idx = random.randint(0, len(correct))
                    correct.insert(idx, wrong)
                    options.append(correct)  
                single_wrong = wrong_answers.copy() + golden_answers.copy()
                while len(options) < 4 and single_wrong:
                    options.append(single_wrong.pop())
                if len(options) < 4:
                    continue
                correct_idx = random.randint(0, 4)
                options.insert(correct_idx, golden_answers)  
            str_options = []
            for option in options:
                if isinstance(option, list):
                    option = ', '.join(option)
                str_options.append(option)      
            msg = {
                "source": item['video_id'],
                "raw_question": item['question'],
                "question": item['question'].split('\n')[0],
                "options": str_options,
                "golden_answer": str(correct_idx + 1)
            }
            result.append(msg)
    save_json_data(target_path, result)



def generate_animal_questions(dict, root_dir, out_dir, num):
    keys = list(dict.keys())
    cnt = 1
    question_data = []
    while cnt <= num:
        select_keys = random.sample(keys,2)
        while True:
            vals = [random.choice(dict[key]) for key in select_keys]
            if vals[0] != vals[1]:
                break
        input_names = [select_keys[0], vals[0], select_keys[1]]
        input_figre_paths = get_figure_path(root_dir=root_dir, dir_name=input_names, sample=3)
        golden_option_path = get_figure_path(root_dir=root_dir, dir_name=vals[1], sample=1)
        wrong_option_names = [name for name in ANIMAL_LIST if name not in dict[select_keys[1]]]
        option_paths = get_figure_path(root_dir=root_dir, dir_name=wrong_option_names, sample=4)
        correct_idx = random.randint(0, 4)
        option_paths.insert(correct_idx, golden_option_path[0])
        copy_figure_path(path_list=input_figre_paths, target_dir=os.path.join(out_dir, str(cnt)))
        copy_figure_path(path_list=option_paths, target_dir=os.path.join(out_dir, str(cnt), 'options'))
        item = {
            "source": str(cnt),
            "question": "Choose the appropriate animal from the options for the fourth figure.",
            "golden_answer": str(correct_idx + 1)
        }
        question_data.append(item)
        cnt += 1
    with open(f"{out_dir}/question.json", "w", encoding="utf-8") as f:
        json.dump(question_data, f, ensure_ascii=False, indent=4)
        
    return 


def generate_art_questions(root_dir, out_dir, num): 
    name_ls = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    cnt = 1
    question_data = []
    while cnt <= num:
        select_name = random.choice(name_ls)
        input_num = random.randint(4,6)
        input_figre_paths = get_figure_path(root_dir=root_dir, dir_name=select_name, sample=input_num)
        wrong_option_names = [name for name in name_ls if name != select_name]
        option_paths = get_figure_path(root_dir=root_dir, dir_name=wrong_option_names, sample=4)
        correct_idx = random.randint(0, 4)
        option_paths.insert(correct_idx, input_figre_paths[-1])
        copy_figure_path(path_list=input_figre_paths[:-1], target_dir=os.path.join(out_dir, str(cnt)))
        copy_figure_path(path_list=option_paths, target_dir=os.path.join(out_dir, str(cnt), 'options'))
        item = {
            "source": str(cnt),
            "question": "The input images are some paintings from the same artist. Please select the work that is most likely to be from this artist from the following options.",
            "golden_answer": str(correct_idx + 1)
        }
        question_data.append(item)
        cnt += 1
    with open(f"{out_dir}/question.json", "w", encoding="utf-8") as f:
        json.dump(question_data, f, ensure_ascii=False, indent=4)
        
    return 
    

def generate_shoes_questions(type, root_dir, out_dir, num):

    name_ls = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]   
    cnt = 1
    question_data = []
    while cnt <= num:
        if type == 'sim':
            select_name = random.choice(name_ls)
            input_num = random.randint(4,6)
            input_figre_paths = get_figure_path(root_dir=root_dir, dir_name=select_name, sample=input_num)
            wrong_option_names = [name for name in name_ls if name != select_name]
            option_paths = get_figure_path(root_dir=root_dir, dir_name=wrong_option_names, sample=4)
            correct_idx = random.randint(0, 4)
            option_paths.insert(correct_idx, input_figre_paths[-1])
            copy_figure_path(path_list=input_figre_paths[:-1], target_dir=os.path.join(out_dir, str(cnt)))
            copy_figure_path(path_list=option_paths, target_dir=os.path.join(out_dir, str(cnt), 'options'))
            question = "My friend already has these shoes and he wants a similar pair. Which of the following shoes in options should I consider giving him?"
        else:
            input_num = random.randint(3,5)
            select_name = random.sample(name_ls, input_num)
            input_figre_paths = get_figure_path(root_dir=root_dir, dir_name=select_name, sample=input_num+4)
            correct_option_names = [name for name in name_ls if name not in select_name]
            correct_option_path = get_figure_path(root_dir=root_dir, dir_name=correct_option_names, sample=1) 
            option_paths = input_figre_paths[-4:]
            correct_idx = random.randint(0, 4)
            option_paths.insert(correct_idx, correct_option_path[0])
            copy_figure_path(path_list=input_figre_paths[:-4], target_dir=os.path.join(out_dir, str(cnt)))
            copy_figure_path(path_list=option_paths, target_dir=os.path.join(out_dir, str(cnt), 'options'))
            question = "My friend already has these shoes, but he likes different shoes. Which of the following shoes in options should I consider giving him?"
            
        item = {
            "source": str(cnt),
            "question": question,
            "golden_answer": str(correct_idx + 1)
        }
        question_data.append(item)
        cnt += 1
    with open(f"{out_dir}/question.json", "w", encoding="utf-8") as f:
        json.dump(question_data, f, ensure_ascii=False, indent=4)
    return 


def generate_plant_questions(root_dir, out_dir, num):
    name_ls = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    type_dic = defaultdict(list)
    for name in name_ls:
        type, _ = name.split("___")
        type_dic[type].append(name)
    cnt = 1
    question_data = []
    while cnt <= num:
        select_name = random.choice(name_ls)
        type = select_name.split("___")[0]
        if len(type_dic[type]) <= 2 or 'healthy' in select_name:
            continue  
        wrong_option_names = [name for name in type_dic[type] if name != select_name]
        input_num = random.randint(4,6)
        input_figre_paths = get_figure_path(root_dir=root_dir, dir_name=select_name, sample=input_num)
        option_paths = get_figure_path(root_dir=root_dir, dir_name=wrong_option_names, sample=4)
        correct_idx = random.randint(0, 4)
        option_paths.insert(correct_idx , input_figre_paths[-1])
        copy_figure_path(path_list=input_figre_paths[:-1], target_dir=os.path.join(out_dir, str(cnt)))
        copy_figure_path(path_list=option_paths, target_dir=os.path.join(out_dir, str(cnt), 'options'))
        item = {
            "source": str(cnt),
            "question": "",
            "golden_answer": str(correct_idx + 1)
        }
        question_data.append(item)
        cnt += 1
    with open(f"{out_dir}/question.json", "w", encoding="utf-8") as f:
        json.dump(question_data, f, ensure_ascii=False, indent=4)
    return 


# 读取 raw_question.json
def generate_recipe_question():
    with open("./raw_question/deductive/recipe/raw_question.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 构造新的数据
    new_data = []
    for item in raw_data['questions']:
        new_item = {
            "source": item["folder"],
            "question": item["question"],
            "golden_answer": item["correct_sequence"]
        }
        new_data.append(new_item)

    # 写入 question.json
    with open("./raw_question/deductive/recipe/question.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print("✅ 已成功生成 question.json")


def generate_crowd_questions(root_path, target_dir, num):
    images = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith(('.jpg'))]  # 获取图片文件
    images = natsort.natsorted(images)
    cnt = 0
    results = []
    output_json_path = os.path.join(target_dir, 'question.json')
    while cnt < num:
        start_idx = random.randint(0,len(images)-1)
        next_idx_start = start_idx
        idx_ls = [start_idx]
        image_count = random.randint(4,6)
        for _ in range(image_count-1):
            idx = random.randint(next_idx_start + 1, next_idx_start + 3)
            idx_ls.append(idx)
            next_idx_start = idx
        if max(idx_ls) > len(images) - 1:
            continue
        path_list = [images[idx] for idx in idx_ls]
        
        shuffled_list = path_list.copy()
        random.shuffle(shuffled_list)
        orders = []  # 存储图片重命名后的顺序与原始顺序的对应关系
        for img in path_list:
            order = str(shuffled_list.index(img) + 1)
            orders.append(order)
        cnt += 1
        copy_figure_path(path_list=shuffled_list, target_dir=os.path.join(target_dir, str(cnt)))
        results.append({'source': str(cnt), 'question':'', 'golden_answer': ('-').join(orders)})
    save_json_data(output_json_path, results)
# Path to the main directory


def convert_and_copy_image(src_path, dst_path):
    """Convert JPG image to PNG and copy it."""
    with Image.open(src_path) as img:
        img.save(dst_path, 'PNG')

def remove_leading_zeros_in_subdirs(path):
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):  # 只处理子目录
            new_name = name.lstrip("0")
            if not new_name:  
                new_name = "0"  # 防止目录名全是 0
            new_path = os.path.join(path, new_name)
            if full_path != new_path:
                print(f"Renaming: {full_path} -> {new_path}")
                os.rename(full_path, new_path)


def generate_sport_questions(main_dir):
    # List all subdirectories
    remove_leading_zeros_in_subdirs(main_dir)
    
    subfolders = [f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))]

    json_data = []

    for folder in subfolders:
        correct_dir = os.path.join(main_dir, folder, 'correct')
        incorrect_dir = os.path.join(main_dir, folder, 'incorrect')
        input_dir = os.path.join(main_dir, folder, 'input')
        
        # Step 1: Move files from input folder to the folder level and rename them as 1.png, 2.png, etc.
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
        for idx, file in enumerate(input_files, start=1):
            old_path = os.path.join(input_dir, file)
            new_name = f"{idx}.png"
            new_path = os.path.join(main_dir, folder, new_name)
            convert_and_copy_image(old_path, new_path)  # Convert and copy as PNG
        
        # Step 2: Merge "correct" and "incorrect" into a new "options" folder
        options_dir = os.path.join(main_dir, folder, 'options')
        os.makedirs(options_dir, exist_ok=True)
        
        correct_files = [f for f in os.listdir(correct_dir) if f.endswith('.jpg')]
        incorrect_files = [f for f in os.listdir(incorrect_dir) if f.endswith('.jpg')]
        
        # Copy correct and incorrect files to "options" folder
        all_files = correct_files + incorrect_files
        random.shuffle(all_files)
        
        # Step 3: Rename the files as 1.png, 2.png, etc., in the options folder
        for idx, file in enumerate(all_files, start=1):
            old_path = os.path.join(correct_dir if file in correct_files else incorrect_dir, file)
            new_name = f"{idx}.png"
            new_path = os.path.join(options_dir, new_name)
            convert_and_copy_image(old_path, new_path)  # Convert and copy as PNG
            
            # Record the source of the correct files
            if file in correct_files:
                json_data.append({
                    'source': folder,
                    'question': "",  # Empty for now, can be filled later
                    'golden answer': idx
                })

        # Clean up: Remove original "correct" and "incorrect" directories
        shutil.rmtree(correct_dir)
        shutil.rmtree(incorrect_dir)
        shutil.rmtree(input_dir)

    # Step 4: Save the json data to a file
    with open(os.path.join(main_dir, 'question.json'), 'w') as f:
        json.dump(json_data, f, indent=4)

def process_video(root_dir, target_dir, start_index=1):

    def sample_frames_from_video(video_path, num_frames):
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频的总帧数和帧率
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 计算需要采样的帧的索引
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # 设置视频的当前帧
            ret, frame = cap.read()  # 读取该帧
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames

    
    index = start_index
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                num_frames = 10
                new_dir = os.path.join(target_dir, filename.split('.')[0])
                video_path = os.path.join(dirpath, filename)
                print(f"Processing video: {video_path}")
                frames = sample_frames_from_video(video_path, num_frames)
                os.makedirs(new_dir)
                for i, frame in enumerate(frames):
                    frame_filename = os.path.join(new_dir, f"{i+1}.png")
                    cv2.imwrite(frame_filename, frame)
                print(f"Extracted {num_frames} frames from {filename}")
                index += 1
                if index > 200:
                    return 

def shuffle_images_in_subdirs(parent_path):
    result = []  # 用于存储所有子目录的结果
    # 遍历输入路径下的所有子目录
    for dir_name in os.listdir(parent_path):
        dir_path = os.path.join(parent_path, dir_name, 'original')
        # 确保是一个目录
        if os.path.isdir(dir_path):
            images = [f for f in os.listdir(dir_path) if f.endswith(('.png'))]  # 获取图片文件
            # images.sort()  # 确保按文件名排序
            images = natsort.natsorted(images)
            # 获取图片的原始顺序（例如：["1", "2", "3", ...]）
            golden_answer = [f.split('.')[0] for f in images]
            
            # 随机打乱图片顺序
            shuffled_images = golden_answer.copy()
            random.shuffle(shuffled_images)

            # 构建重命名后的图片顺序与原始顺序的映射
            order = []  # 存储图片重命名后的顺序与原始顺序的对应关系
            for i, img in enumerate(images):
                new_name = shuffled_images[i] + os.path.splitext(img)[1]
                original_path = os.path.join(dir_path, img)
                new_path = os.path.join(parent_path, dir_name, new_name)

                # 记录重命名后图片的顺序
                order.append(shuffled_images[i])
                # 复制文件到新目录（不删除原文件）
                shutil.copy2(new_path, original_path)
                
            order = '-'.join(order)
            # 保存结果
            result.append({
                "source": dir_name,
                "golden answer": order  # 保存原始顺序
            })

    # 将结果保存为 json 文件
    json_file = os.path.join(parent_path, 'question.json')
    with open(json_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Shuffling complete. The results are saved in {json_file}")


def generate_driving_data(origin_path, output_dir):
    dir_cnt = 1
    for dir_name in os.listdir(origin_path):
        input_dir = os.path.join(origin_path, dir_name)
        
        jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
        jpg_files = natsort.natsorted(jpg_files)
        group_size = 10
        # 按每10个文件分组
        for i in range(0, len(jpg_files), group_size):
            group = jpg_files[i:i + group_size]
            
            # 创建每组的子目录
            group_dir = os.path.join(output_dir, f'{dir_cnt}')
            if not os.path.exists(group_dir):
                os.makedirs(group_dir)
            dir_cnt += 1
            # 将每个文件转换为 PNG 并保存
            cnt = 1
            for file in group:
                jpg_path = os.path.join(input_dir, file)
                png_path = os.path.join(group_dir, f'{cnt}.png')
                
                # 打开图像并转换为 PNG 格式
                with Image.open(jpg_path) as img:
                    img.save(png_path, 'PNG')
                cnt += 1

                print(f'Converted {file} to {png_path}')    
    

def generate_TVbench_question():
    raw_question_path = './raw_data/TVbench/action_localization/action_localization.json'
    target_dir = './raw_question/temporal/TVbench/'
    data = load_json_data(raw_question_path)
    count = 1
    result = []
    for item in data:
        video = item.get("video", "")
        video_name = video.split('.')[0]
        old_path = os.path.join(target_dir, video_name)
        new_path = os.path.join(target_dir, str(count))
        os.rename(old_path, new_path)
        msg = {
            'source': str(count),
            'question': item['question'],
            'golden_answer': item['answer']
        }
        result.append(msg)
        count += 1
    save_json_data(os.path.join(target_dir,'question.json'), result)   


def process_images_in_driving(base_path):
    # 存储所有处理过的目录信息
    processed_data = []

    # 遍历当前路径下的每个子目录
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        
        # 跳过非文件夹
        if not os.path.isdir(subdir_path):
            continue

        # 获取子目录下所有图片文件
        images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png'))]    
        images = natsort.natsorted(images)
        # 跳过图片数少于8的文件夹
        if len(images) < 8:
            continue
        
        sampled_images = []
        n = len(images)
        possible_images = images[:n-1]  # 排除最后一张图片
        sampled_images = random.sample(possible_images, n - 5) 
        sampled_images = natsort.natsorted(sampled_images)
        # 4. 将未被采样的5个图重新打乱
        remaining_images = [img for img in images if img not in sampled_images]  # 剩余的5张图
        random.shuffle(remaining_images)

        # 选择最后一张图的位置m
        last_image = sampled_images[-1] if sampled_images else None
        index = images.index(last_image) + 1
        target_index = remaining_images.index(f'{index+1}.png') + 1

        # 5. 在option dir下创建目录
        option_dir = os.path.join(subdir_path, "options")
        if not os.path.exists(option_dir):
            os.makedirs(option_dir)

        # 6. 重命名未采样的图片为1-5 png，转移到新目录
        for idx, img in enumerate(remaining_images):
            old_path = os.path.join(subdir_path, img)
            new_name = f"{idx+1}.png"
            new_path = os.path.join(option_dir, new_name)
            shutil.move(old_path, new_path)

        # 7. 重命名采样的图片为1, 2, 3... png
        for idx, img in enumerate(sampled_images):
            old_path = os.path.join(subdir_path, img)
            new_name = f"0{idx+1}.png"  # 从6开始命名
            new_path = os.path.join(subdir_path, new_name)
            shutil.move(old_path, new_path)
            
        for idx in range(1, len(sampled_images)+1):
            old_name = os.path.join(subdir_path, f"0{idx}.png")
            new_name = os.path.join(subdir_path, f"{idx}.png")
            os.rename(old_name, new_name)

        # 8. 生成JSON文件的条目
        item = {
            'source': subdir,
            'question': 'Which image in the options is most likely to occur at the next moment?',
            'golden_answer': str(target_index)
        }
        processed_data.append(item)

    # 保存为JSON文件
    with open(os.path.join(base_path, 'question.json'), 'w') as json_file:
        json.dump(processed_data, json_file, indent=4)

def generate_bird_questions(root_path):
    results = []
    for subdir, dirs, files in os.walk(root_path):
        if 'options' in dirs:
            options_path = os.path.join(subdir, 'options')
            
            # 列出该目录下所有的文件
            option_files = os.listdir(options_path)
            
            # 确保 1.png 存在于该目录下
            if '1.png' in option_files and '2.png' not in option_files:
                # 随机生成一个数字k，假设k为1至5之间的一个数字
                k = random.randint(1, 5)
                # 重命名 1.png 文件
                new_name = f'{k}.png'
                os.rename(os.path.join(options_path, '1.png'), os.path.join(options_path, new_name))
                
                # 剩下的4个文件重命名
                remaining_files = [f for f in option_files if f != '1.png']
                remaining_numbers = [n for n in range(1, 6) if n != k]
                random.shuffle(remaining_numbers)
                
                # 重命名剩下的文件
                for i, file in enumerate(remaining_files):
                    new_name = f'{remaining_numbers[i]}.png'
                    os.rename(os.path.join(options_path, file), os.path.join(options_path, new_name))
                
                results.append({
                    'source': os.path.basename(subdir),
                    'question': '',
                    'golden_answer': str(k)
                })
                
    # 将结果保存为 JSON 文件
    with open(os.path.join(root_path, 'question_1.json'), 'w') as json_file:
        json.dump(results, json_file, indent=4)



parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='animal')
args = parser.parse_args() 


if __name__ == '__main__':
    if args.task == 'animal':
        rel_path = '../../raw_data/animal/rel.json'
        with open(rel_path, 'r') as f:
            rel_dic = json.load(f)
        eat_dic = rel_dic['catch_and_feed']
        comp_dic = rel_dic['competition']
        generate_animal_questions(dict=eat_dic, root_dir='../../raw_data/animal/', out_dir='./predatory', num=200)
        # generate_animal_questions(dict=comp_dic, root_dir='./animal/raw_data/', out_dir='./animal/comp', num=200)
    elif args.task == 'art':
        generate_art_questions(root_dir='./raw_data/art/', out_dir='./raw_question/analogical/art/', num=200)
    elif args.task == 'shoes':
        generate_shoes_questions(type='sim', root_dir='./raw_data/shoes/', out_dir='./raw_question/analogical/sim_shoes/', num=100)
        generate_shoes_questions(type='dif', root_dir='./raw_data/shoes/', out_dir='./raw_question/analogical/dif_shoes/', num=100)
    elif args.task == 'plant':
        generate_plant_questions(root_dir='./raw_data/plant_disease/', out_dir='./raw_question/inductive/plant/', num=200)
    elif args.task == 'fluid':
        generate_fluid_questions('./raw_question/causal/fluid/question.json')
    elif args.task == 'bird':
        generate_bird_questions('./raw_question/inductive/bird/')
    elif args.task == 'sport':
        generate_sport_questions('./raw_question/inductive/sport/')
    elif args.task == 'poke':
        generate_poke_questions('./raw_question/deductive/poke/question_with_option.json')
    elif args.task == 'crowd':
        generate_crowd_questions(root_path='./raw_data/crowd/', target_dir='./raw_question/temporal/crowd/', num=200)