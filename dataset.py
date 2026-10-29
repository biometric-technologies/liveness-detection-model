import cv2
import os
import shutil
import zlib
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, default=None, help="path to root 'data' folder")
parser.add_argument("-r", "--resized_dir", type=str, default=None, help="path to root resized-data folder")
parser.add_argument("-s", "--target_size", type=int, default=224, help="resized image is square; target size-side size")

def resize_image_with_aspect_ratio(image, target_size=224, padding_color=(0, 0, 0)):
    current_height, current_width = image.shape[:2]
    aspect_ratio = current_width / float(current_height)
    if aspect_ratio > 1:  # Horizontal image
        new_width = target_size
        new_height = int(new_width / aspect_ratio)

        pad_height = target_size - new_height
        top_pad = pad_height // 2
        padded_image = np.full((target_size, new_width, 3), padding_color, dtype=np.uint8)

        resized_image = cv2.resize(image, (new_width, new_height))
        padded_image[top_pad:top_pad + new_height, :, :] = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    else:  # Vertical image
        new_height = target_size
        new_width = int(new_height * aspect_ratio)

        pad_width = target_size - new_width
        left_pad = pad_width // 2
        padded_image = np.full((new_height, target_size, 3), padding_color, dtype=np.uint8)

        resized_image = cv2.resize(image, (new_width, new_height))
        padded_image[:, left_pad:left_pad + new_width, :] = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    args = vars(parser.parse_args())

    data_dir = args["input_dir"]
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")
    resized_dir = args["resized_dir"]
    target_size = args["target_size"]

    # CelebA-Spoof keeps each subject in a numbered folder with many frames.
    # The validation split is made by subject, so that no person appears in both
    # train and validation (a split by file would leak identities and overstate accuracy).
    for dataset_split in ['train', 'test']:
        split_dir = os.path.join(data_dir, dataset_split)
        for label in ['real', 'spoof']:
            os.makedirs(os.path.join(split_dir, label), exist_ok=True)
            if dataset_split == 'train':
                os.makedirs(os.path.join(valid_dir, label), exist_ok=True)

        numbered_folders = sorted(f for f in os.listdir(split_dir)
                                  if os.path.isdir(os.path.join(split_dir, f)) and f not in ('real', 'spoof'))

        for numbered_folder in numbered_folders:
            numbered_folder_path = os.path.join(split_dir, numbered_folder)
            # deterministic 20% of subjects go to validation
            to_valid = dataset_split == 'train' and zlib.crc32(numbered_folder.encode()) % 5 == 0
            target_root = valid_dir if to_valid else split_dir

            for sub_folder in ['live', 'spoof']:
                sub_folder_path = os.path.join(numbered_folder_path, sub_folder)
                label = 'real' if sub_folder == 'live' else 'spoof'
                if os.path.exists(sub_folder_path):
                    for image_file in os.listdir(sub_folder_path):
                        if image_file.lower().endswith(('.png', '.jpg')):
                            shutil.move(os.path.join(sub_folder_path, image_file),
                                        os.path.join(target_root, label, f"{numbered_folder}_{image_file}"))
            shutil.rmtree(numbered_folder_path)

    for sub_dir in os.listdir(data_dir):
        split_dir = os.path.join(data_dir, sub_dir)
        new_dir = os.path.join(resized_dir, sub_dir)
        for sub_folder in ['real', 'spoof']:
            sub_folder_path = os.path.join(split_dir, sub_folder)
            out_dir = os.path.join(new_dir, sub_folder)
            os.makedirs(out_dir, exist_ok=True)
            for img in os.listdir(sub_folder_path):
                image_path = os.path.join(sub_folder_path, img)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                resized_image = resize_image_with_aspect_ratio(image, target_size)
                cv2.imwrite(os.path.join(out_dir, img), resized_image)
