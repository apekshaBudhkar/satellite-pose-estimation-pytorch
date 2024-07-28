import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from pathlib import Path
import json
import cv2

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])


def apply_agc(image):
    min = np.min(image)
    max = np.max(image)
    span = max - min
    span = span if (span > 0.0) else 1.0

    result = (image - min) * (255.0 / span)
    np.clip(result, 0, 255, result)
    return result.astype("uint8")


def get_buffer(path):
    with open(path, mode="rb") as f:
        for b in f:
            np_arr = np.frombuffer(b, dtype=np.float32)
    return np_arr


class InfraredPoseEstimationDS(Dataset):
    """
    Data loader for the satellite infrared pose estimation dataset
    """

    def __init__(self, data_root, transform=None):
        self.root = data_root
        self.ground_truth_path = Path.joinpath(Path(data_root), "poses.json")
        self.json_data = self.__load_json()

        self.dataset_ids = self.__extract_ids_from_json()
        self.img_size = (240, 320)
        self.transform = transform

    def __len__(self):
        return len(self.dataset_ids)

    def __getitem__(self, idx):
        id = self.dataset_ids[idx]
        filename = self.__get_image_filename(idx)
        input = Image.open(filename)

        if self.transform is not None:
            input = self.transform(input)

        data_point = self.json_data[id]
        output = self.__get_pose_value(data_point)

        return id, input, torch.tensor(output)

    def __get_image_filename(self, idx) -> str:
        id = self.dataset_ids[idx]
        filepath = Path.joinpath(Path(self.root), f"{id}.png")
        return str(filepath)

    def __load_json(self):
        with open(self.ground_truth_path, "r") as f:
            return json.load(f)

    def __get_pose_value(self, data):
        return (data["x"], data["y"], data["yaw"])

    def __extract_ids_from_json(self):
        data = self.__load_json()
        ids = []
        for key in data:
            ids.append(key)
        return ids


class InfraredPoseEstimationDSSimplified(Dataset):
    """
    Data loader for the satellite infrared pose estimation dataset
    """

    def __init__(self, data_root, split, transform=None, load_as_single_channel=True):
        self.root = data_root
        if split not in ["train", "test", "validation"]:
            raise ValueError(f"Unknown split type: {split}")
        self.split = split
        self.input_directory = Path.joinpath(Path(data_root), split, "images")

        json_directory = Path.joinpath(Path(data_root), split, f"{split}.json")
        self.poses = self.__load_json(json_directory)

        self.dataset_ids = os.listdir(self.input_directory)
        self.transform = transform
        self.single_channel = load_as_single_channel

    def __len__(self):
        return len(self.dataset_ids)

    def __getitem__(self, idx):
        id = self.dataset_ids[idx]
        img_filepath = Path.joinpath(Path(self.input_directory), id)

        if self.single_channel == False:
            img = np.array(Image.open(img_filepath))
            img = Image.fromarray(img).convert("RGB")

        else:
            img = Image.open(img_filepath)
        x = img

        if self.transform is not None:
            x = self.transform(x)
        pose_json = self.poses[id]
        yaw = pose_json["yaw"]
        output = np.array([pose_json["x"], pose_json["y"], yaw])

        return id, x, torch.tensor(output)

    def __load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)
