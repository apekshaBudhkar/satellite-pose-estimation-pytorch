import click
import json
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt


paths = {
    "yolo10x10": "/home/apeksha/Projects/masters/satellite-pose-estimation-infrared/outputs/epoch40_yolo10_2024-3-2_14-36.json",
    "yolo7x7": "/home/apeksha/Projects/masters/satellite-pose-estimation-infrared/outputs/epoch45_yolo7_2024-3-2_15-13.json",
    "yolo5x5": "/home/apeksha/Projects/masters/satellite-pose-estimation-infrared/outputs/epoch40_yolo5_2024-3-1_12-46.json",
    "resnet18": "/home/apeksha/Projects/masters/satellite-pose-estimation-infrared/outputs/epoch50_resnet_2024-3-2_16-29.json",
}


def load_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def get_loss_data_as_list(json_data, idx):
    outputs = [x["losses"][idx] for x in json_data]
    return list(outputs)


@click.command()
@click.option("-o", "--output_dir", required=False, default="outputs")
def main(output_dir):
    num_parameters = []
    pos_loss = []
    ori_loss = []

    for path in paths:
        json_data = load_json(paths[path])
        summ = json_data["model_summary"]
        num_parameters.append(summ["params"] )
        pos_loss.append(summ["med_pos_loss"])
        ori_loss.append(summ["med_ori_loss"])

    plt.figure()
    plt.scatter(num_parameters, pos_loss, color="black", label="Position Loss")
    plt.scatter(num_parameters, ori_loss, color="red", label="Orientation Loss")

    list_of_models = list(paths.keys())
    for i in range(0, len(list_of_models)):
        model_name = list_of_models[i]
        print(model_name)
        txt = model_name
        plt.annotate(
            txt,
            (num_parameters[i], pos_loss[i]),
            textcoords="offset points",
            xytext=(5, 10),
            ha="center",
        )
        plt.annotate(
            txt,
            (num_parameters[i], ori_loss[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    

    plt.title("Overall loss vs. number of parameters")
    plt.legend()
    plt.xlabel("Number of model parameters (millions)")
    plt.ylabel("Loss values")
    # plt.xscale("log")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
