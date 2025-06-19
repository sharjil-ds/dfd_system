import os
import shutil
import random
from tqdm import tqdm


def collect_image_paths(base_dir):
    image_paths = []
    for label in ['real', 'fake']:
        label_dir = os.path.join(base_dir, label)
        for video_folder in os.listdir(label_dir):
            video_path = os.path.join(label_dir, video_folder)
            if not os.path.isdir(video_path):
                continue
            for img_file in os.listdir(video_path):
                if img_file.lower().endswith('.jpg'):
                    full_path = os.path.join(video_path, img_file)
                    image_paths.append((full_path, label))
    return image_paths


def split_and_copy(images, output_base_dir, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(images)
    total = len(images)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split_name, split_images in splits.items():
        for img_path, label in tqdm(split_images, desc=f"Copying to {split_name}"):
            base_name = os.path.basename(os.path.dirname(img_path)) + "_" + os.path.basename(img_path)
            dst_dir = os.path.join(output_base_dir, split_name, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, base_name)
            shutil.copy2(img_path, dst_path)


if __name__ == "__main__":
    input_dir = "cropped_faces"
    output_dir = "dataset_faces"

    images = collect_image_paths(input_dir)
    split_and_copy(images, output_dir)
