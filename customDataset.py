import os
import shutil
import random

RAW_DIR = "Dataset"
OUTPUT_DIR = "./sdnet_flat_dataset"
CLASS_SIZE = 3000


def create_flat_dataset():
    cracked_images = []
    noncracked_images = []

    if not os.path.exists(RAW_DIR):
        print(f"ERROR: Directory '{RAW_DIR}' does not exist.")
        return

    print(f"Scanning '{RAW_DIR}' for images...")

    # Recursively gather image file paths
    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                path_parts = [part.upper() for part in root.split(os.sep)]

                is_cracked = any(part.startswith('C') for part in path_parts)
                is_uncracked = any(part.startswith(('U', 'N')) for part in path_parts)

                if is_cracked:
                    cracked_images.append(full_path)
                elif is_uncracked:
                    noncracked_images.append(full_path)

    print(f"Found {len(cracked_images)} raw cracked images.")
    print(f"Found {len(noncracked_images)} raw non-cracked images.")

    if len(cracked_images) < CLASS_SIZE or len(noncracked_images) < CLASS_SIZE:
        print(f"ERROR: One of the classes has fewer than {CLASS_SIZE} images.")
        return

    # Randomly sample exactly CLASS_SIZE per category
    random.seed(42)
    cracked_sampled = random.sample(cracked_images, CLASS_SIZE)
    noncracked_sampled = random.sample(noncracked_images, CLASS_SIZE)

    # Recreate output folder
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # Copy files directly into their respective class folder
    for class_name, img_list in [('cracked', cracked_sampled), ('noncracked', noncracked_sampled)]:
        dest_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        for src_path in img_list:
            shutil.copy(src_path, os.path.join(dest_dir, os.path.basename(src_path)))

    print(f"\nSUCCESS: Flat dataset generated in '{OUTPUT_DIR}'!")
    print(f"- {OUTPUT_DIR}/cracked/ ({len(cracked_sampled)} images)")
    print(f"- {OUTPUT_DIR}/noncracked/ ({len(noncracked_sampled)} images)")


if __name__ == '__main__':
    create_flat_dataset()