import click
import json
from pathlib import Path
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


paths = {
    "yolo10x10": "/home/apeksha/Projects/masters/satellite-pose-estimation-infrared/outputs/apr7/epoch35_yolo10_2024-4-7_1-24.json",
    "yolo7x7": "/home/apeksha/Projects/masters/satellite-pose-estimation-infrared/outputs/apr7/epoch35_yolo7_2024-4-7_2-38.json",
    "yolo5x5": "/home/apeksha/Projects/masters/satellite-pose-estimation-infrared/outputs/apr7/epoch35_yolo5_2024-4-7_2-1.json",
    "resnet18": "/home/apeksha/Projects/masters/satellite-pose-estimation-infrared/outputs/apr7/epoch35_resnet_2024-4-7_4-4.json",
}


def load_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def get_loss_data_as_list(json_data, idx):
    outputs = [x["losses"][idx] for x in json_data]
    return list(outputs)


def create_box_plot(data, title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    times_new_roman = {"fontname": "Times New Roman",
                       'size'   : 17}
    b = sns.boxplot(
        data=data,
        x="Network",
        y="Loss",
        palette=["#8a0101", "#857c78", "#FC1A05", "#04080f", "#331832"],
        showfliers=False,
    )

    b.axes.set_xlabel("Network", **times_new_roman)
    b.axes.set_ylabel("Error Distribution", **times_new_roman)

    plt.savefig(f"{title}.png", bbox_inches="tight", dpi=500)
    plt.show()


def main():
    pos_loss_data = {}
    for key in paths:
        data = load_json(paths[key])["model_performance"]
        pos_loss_data[key] = get_loss_data_as_list(data, 0)

    orientation_loss_data = {}
    for key in paths:
        data = load_json(paths[key])["model_performance"]
        orientation_loss_data[key] = get_loss_data_as_list(data, 1)

    pos_data = pd.DataFrame(
        [
            (network, loss)
            for network, losses in pos_loss_data.items()
            for loss in losses
        ],
        columns=["Network", "Loss"],
    )

    ori_data = pd.DataFrame(
        [
            (network, loss)
            for network, losses in pos_loss_data.items()
            for loss in losses
        ],
        columns=["Network", "Loss"],
    )

    create_box_plot(pos_data, "Position Loss Distribution")
    create_box_plot(ori_data, "Orientation Loss Distribution")


if __name__ == "__main__":
    main()
