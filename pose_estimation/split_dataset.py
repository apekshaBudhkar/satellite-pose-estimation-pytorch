import torch
from torch.utils.data import random_split

from dataloaders.dataloader_infrared import InfraredPoseEstimationDS

import matplotlib.pyplot as plt
import cv2
import numpy as np

from pathlib import Path
import json
import os
import shutil

"""
This script takes the raw dataset collected from the lab, and splits it evenly 
into a train/validation/test split.
"""
SOURCE = "/vivado/spot_data/2024/experimentv5/combined"
output_root = "/vivado/spot_data/2024/experimentv5"


def save_data_to_disk(dataloader, split):
    x_pos = []
    y_pos = []
    yaw = []

    output_json_path = Path(Path(output_root), split, f"{split}.json")
    output_json = {}
    images_directory = Path(Path(output_root), split, "images")
    os.mkdir(images_directory)

    count = 0

    for id, img, pose in dataloader:
        if (pose[0].item() < 1) or (pose[1].item() < -0.55 or pose[1].item() > 0.55):
            continue

        img_name = f"{id}.png"

        output_img_path = images_directory.joinpath(img_name)
        cv2.imwrite(str(output_img_path), np.array(img))

        output_json[img_name] = {
            "x": pose[0].item(),
            "y": pose[1].item(),
            "yaw": pose[2].item(),
        }

        x_pos.append(pose[0])
        y_pos.append(pose[1])
        yaw.append(pose[2])
        count += 1
        if count % 500 == 0:
            print(count)
    fig = plt.figure()

    plt.subplot(1, 3, 1)
    plt.hist(x_pos, density=False, color="#a20025", edgecolor="white")
    plt.xlabel("X distances")
    plt.ylabel("Distribution")

    plt.subplot(1, 3, 2)
    plt.title(split)
    plt.hist(y_pos, density=False, color="#a20025", edgecolor="white")
    plt.xlabel("Y distances")
    plt.ylabel("Distribution")

    plt.subplot(1, 3, 3)
    plt.hist(yaw, density=False, color="#a20025", edgecolor="white")
    plt.xlabel("Yaw angles")
    plt.ylabel("Distribution")
    print(f"Saving to disk {split}")
    plt.savefig(f"{split}.png")

    with open(output_json_path, "w") as f:
        json.dump(output_json, f)


def main():
    generator = torch.Generator().manual_seed(42)
    dataloader = InfraredPoseEstimationDS(SOURCE)
    train, validation, test = random_split(dataloader, [0.80, 0.05, 0.15], generator)

    os.mkdir(f"{output_root}/train")
    os.mkdir(f"{output_root}/test")
    os.mkdir(f"{output_root}/validation")

    print(len(train))
    save_data_to_disk(train, "train")

    print(len(test))
    save_data_to_disk(test, "test")

    print(len(validation))
    save_data_to_disk(validation, "validation")

    # Extra step to copy the raw .txt files.
    # We're doing it so the data analysis is easier after.
    raw_folder = Path(output_root).joinpath("test").joinpath("raw")
    os.mkdir(raw_folder)
    for txt_file in Path(SOURCE).glob("*.txt"):
        destination = Path(raw_folder).joinpath(txt_file.name)
        shutil.copy(txt_file, destination)


if __name__ == "__main__":
    main()
