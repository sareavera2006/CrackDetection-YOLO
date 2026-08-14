import os
import shutil
import random
from tqdm import tqdm
import cv2

RAW_DIR = "Dataset"
CLAHE_DIR = "./dataset_clahe"
YOLO_DIR = "./sdnet_yolo_cls"

# Ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

CLASS_SIZE = 5000

# CLAHE Initialization
clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(8,8))

def applyCLAHE(src, dest):

    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    enhancedL = clahe.apply(l)
    enhancedLab = cv2.merge([enhancedL, a, b])
    enhancedImage = cv2.cvtColor(enhancedLab, cv2.COLOR_LAB2BGR)

    cv2.imwrite(dest, enhancedImage)

def preparation(srcPath):
    cracked_images = []
    noncracked_images = []

    items = os.listdir(RAW_DIR)
    print(f'Total Items {len(items)}')

    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)

                path_parts = [part.upper() for part in root.split(os.sep)]

                is_cracked = any(part.startswith('C') for part in path_parts)
                is_uncracked = any(part.startswith('N') for part in path_parts)

                if is_cracked:
                    cracked_images.append(full_path)
                elif is_uncracked:
                    noncracked_images.append(full_path)

    print(f"Raw Number of {len(cracked_images)} cracked images.")
    print(f"Raw Number of {len(noncracked_images)} non-cracked images")

    if len(cracked_images) == 0 and len(noncracked_images) == 0:
        print("ERROR: No images were found.")
        return

    print("Starting Dataset Compilation for YOLO")

    # Ensures the amount between cracked and noncracked are balanced
    random.seed(42)
    cracked_sampled = random.sample(cracked_images, CLASS_SIZE)
    noncracked_sampled = random.sample(noncracked_images, CLASS_SIZE)

    if os.path.exists(srcPath):
        shutil.rmtree(srcPath)

    for class_name, img_list in [('cracked', cracked_sampled), ('noncracked', noncracked_sampled)]:
        random.shuffle(img_list)

        train_end = int(len(img_list) * TRAIN_RATIO) # Extracts 80% of the dataset
        val_end = train_end + int(len(img_list) * VAL_RATIO) # Extracts the 10% and adds it to the 80%

        train_files = img_list[:train_end]
        val_files = img_list[train_end:val_end]
        test_files = img_list[val_end:]

        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            dest_dir = os.path.join(srcPath, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            # NON-CLAHE
            if srcPath == YOLO_DIR:
                for src_path in tqdm(files, desc=f"Processing {class_name} ({split})"):
                    shutil.copy(src_path, os.path.join(dest_dir, os.path.basename(src_path)))

            # CLAHE
            if srcPath == CLAHE_DIR:
                for src_path in tqdm(files, desc=f"Processing {class_name} ({split})"):
                    dest_path = os.path.join(dest_dir, os.path.basename(src_path))
                    applyCLAHE(src_path, dest_path)

    train_count = int(CLASS_SIZE * TRAIN_RATIO)
    val_count = int(CLASS_SIZE * VAL_RATIO)
    test_count = CLASS_SIZE - (train_count + val_count)

    print(f"Dataset successfully structured in '{srcPath}'")
    print(f"Train split: {train_count} images per class")
    print(f"Val split: {val_count} images per class")
    print(f"Test split: {test_count}")

if __name__ == '__main__':
    preparation(CLAHE_DIR)

