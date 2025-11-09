import os
import natsort
from PIL import Image

def rename_images_in_subdirs(root_dir):
    # 遍历根目录下的所有子目录
    for subdir, _, files in os.walk(root_dir):
        # 筛选出图片文件（根据扩展名）
        images = [f for f in files if f.lower().endswith(('.png'))]
        # 按原始文件名排序
        images = natsort.natsorted(images)
        
        # 遍历并重命名
        for idx, filename in enumerate(images, start=1):
            ext = os.path.splitext(filename)[1]  # 获取原扩展名
            new_name = f"{idx}{ext}"
            old_path = os.path.join(subdir, filename)
            new_path = os.path.join(subdir, new_name)
            if os.path.exists(new_path):
                continue
            
            os.rename(old_path, new_path)
            print(f"重命名: {old_path} -> {new_path}")

def rename_folder(path):
    # 遍历文件夹
    for folder_name in os.listdir(path):
        # 检查文件夹是否以 "video_" 开头
        if folder_name.startswith("video_"):
            # 提取文件夹名中最后一部分数字并转为整数
            new_name = folder_name.split("_")[-1].lstrip("0")  # 去除前导零
            if not new_name:  # 如果全是零，保持为 '0'
                new_name = "0"
            
            # 获取完整路径
            old_folder_path = os.path.join(path, folder_name)
            new_folder_path = os.path.join(path, new_name)
            
            # 重命名文件夹
            os.rename(old_folder_path, new_folder_path)

        print(f"Renamed: {old_folder_path} -> {new_folder_path}")


def resize_for_vlm(path, max_side=1024):
    with Image.open(path) as img:
        img = img.convert("RGBA")  # 保留透明度
        w, h = img.size

        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
            img.save(path, "PNG")
            print(f"Resized {path}: {w}x{h} → {new_size[0]}x{new_size[1]}")
        else:
            print(f"Skipped {path}, already {w}x{h}")

def walk_and_resize(root, max_side=1024):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".png"):
                fp = os.path.join(dp, fn)
                resize_for_vlm(fp, max_side=max_side)


# 使用方法：指定你的目录路径
if __name__ == "__main__":
    root_directory = "/mnt/userdata/ljc/code/RIME/images/abductive/Character Interaction Attribution/34"  # 修改为你要处理的目录路径
    # rename_images_in_subdirs(root_directory)
    walk_and_resize(root_directory)