import click
import json
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import re

import numpy as np


MODEL_OUTPUT_PATH = "/home/apeksha/Projects/masters/satellite-pose-estimation-infrared/outputs/apr1/epoch35_yolo10_2024-3-31_21-37.json"

def load_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def plot_results(actual, pred, elapsed_time, title, ax):
    ax.scatter(elapsed_time, actual, color="black", label="actual", marker="x")
    ax.scatter(elapsed_time, pred, color="red", label="prediction", marker=".")
    ax.set_xlabel("Elapsed time (ms)")
    ax.set_ylabel(title)


def sort_data(model_outputs):
    data = sorted(model_outputs, key=lambda x: x["id"], reverse=False)
    return data


def _get_id(input_str):
    print(input_str[0])
    output = re.findall(r"\d+", input_str[0])
    print(output)
    return output[0]


def get_plot_arrays(data):
    x_real_arr = []
    y_real_arr = []
    yaw_real_arr = []

    x_pred_arr = []
    y_pred_arr = []
    yaw_pred_arr = []

    time_arr = []
    net_time = 0

    for i, data_pt in enumerate(data):
        # if data_pt["elapsed_time"] > 1000:
        #     continue
        # net_time += data_pt["elapsed_time"]
        # id = int(_get_id(data_pt["id"]))
        # if id < 1711065309847 or id > 1711065542020:
        #     continue
        # if net_time < 1980 or net_time > 2241:
        #     continue
        # print(i, end=", ")

        actual_x, actual_y, actual_yaw = data_pt["actual"]
        pred_x, pred_y, pred_yaw = data_pt["predictions"]
        x_real_arr.append(actual_x)
        y_real_arr.append(actual_y)

        yaw_real_arr.append(np.degrees(actual_yaw))

        x_pred_arr.append(pred_x)
        y_pred_arr.append(pred_y)
        yaw_pred_arr.append(np.degrees(pred_yaw))
        net_time += i
        time_arr.append(net_time)

    ground_truth = {
        "x": x_real_arr,
        "y": y_real_arr,
        "yaw": yaw_real_arr,
    }

    predictions = {
        "x": x_pred_arr,
        "y": y_pred_arr,
        "yaw": yaw_pred_arr,
    }
    return ground_truth, predictions, time_arr


def get_plot(ground_truth, pred, time, title):
    fig = plt.figure(figsize=(5, 10))
    plt.suptitle(title)

    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
    ax0 = plt.subplot(gs[0])
    plot_results(ground_truth["x"], pred["x"], time, "X values (m)", ax0)
    plt.grid(axis="x")
    plt.ylim(0.5, 3.0)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.3, 0.62, 0.35, 0.3),
        ncols=2,
    )

    ax1 = plt.subplot(gs[1])
    plot_results(ground_truth["y"], pred["y"], time, "Y values (m)", ax1)
    plt.grid(axis="x")
    plt.ylim(-1, 1)

    ax2 = plt.subplot(gs[2])
    plot_results(ground_truth["yaw"], pred["yaw"], time, "Yaw values ()", ax2)
    plt.grid(axis="x")
    plt.ylim(-180, 180)
    return fig


@click.command()
@click.option("-i", "--input", required=False, default=MODEL_OUTPUT_PATH)
def main(input):
    data = load_json(input)["model_performance"]
    data = sort_data(data)
    # traj_1 = data[0:620]
    # ground_truth, pred, time = get_plot_arrays(traj_1)
    # fig = get_plot(
    #     ground_truth, pred, time, "Satellite trajectories over time (varying y)"
    # )
    # fig.show()
    # plt.show()

    # traj_2 = data[3000:3117]  # Trajectory 2
    # ground_truth, pred, time = get_plot_arrays(traj_2)
    # fig = get_plot(
    #     ground_truth, pred, time, "Satellite trajectories over time (varying yaw)"
    # )
    # fig.show()
    # plt.show()

    # traj_3 = data[5300:5400]  # Trajectory 3
    # ground_truth, pred, time = get_plot_arrays(traj_3)
    # fig = get_plot(
    #     ground_truth, pred, time, "Satellite trajectories over time (varying x)"
    # )
    # fig.show()
    # plt.show()

    # traj_4 = data[19:228]  # Trajectory 4
    # ground_truth, pred, time = get_plot_arrays(traj_4)
    # fig = get_plot(
    #     ground_truth, pred, time, "Satellite trajectories over time (varying x and yaw)"
    # )
    # fig.show()
    # plt.show()

    # traj_4 = data[19:228]  # Trajectory 5
    print(len(data))
    ground_truth, pred, time = get_plot_arrays(data[0:1000])
    fig = get_plot(
        ground_truth, pred, time, "Satellite trajectories over time (varying y and yaw)"
    )
    fig.show()
    plt.show()


if __name__ == "__main__":
    main()
