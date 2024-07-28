import numpy as np
import click
import matplotlib
from matplotlib import pyplot as plt

ROOT = "/home/apeksha/Projects/masters/satellite-pose-estimation-infrared/"
times_new_roman = {"fontname": "Times New Roman", "size": 14}

files_to_plot = {
    "resnet18": f"{ROOT}/scripts/checkpoints/apr7/resnet_2024-4-7_2-43/resnet.npz",
    "yolo5": f"{ROOT}/scripts/checkpoints/apr7/yolo5_2024-4-7_1-29/yolo5.npz",
    "yolo10": f"{ROOT}/scripts/checkpoints/apr7/yolo10_2024-4-7_0-48/yolo10.npz",
    "yolo7": f"{ROOT}/scripts/checkpoints/apr7/yolo7_2024-4-7_2-6/yolo7.npz",
}


styles_dict: dict = {
    "train_loss_pos": {"style": "solid", "color": "black"},
    "train_loss_ori": {"style": "solid", "color": "blue"},
    "val_loss_pos": {"style": "dashed", "color": "black"},
    "val_loss_ori": {"style": "dashed", "color": "blue"},
}


def _plot_variable(plt, array_of_vals, title, style, color):
    x = range(0, len(array_of_vals))
    plt.plot(
        x, array_of_vals, label=title, linestyle=style, color=color
    )
    return plt


def plot_training_loss(numpy_array_path, id):
    np_data = np.load(numpy_array_path)

    plt.clf()
    plot = plt.axes()
    plot = _plot_variable(
        plot,
        np_data["train_loss_pos"],
        "Training loss: position (m)",
        "solid",
        "red",
    )
    plot = _plot_variable(
        plot,
        np_data["train_loss_ori"],
        "Training loss: orientation (rad)",
        "solid",
        "black",
    )
    plot = _plot_variable(
        plot,
        np_data["val_loss_pos"],
        "Val loss: position (m)",
        "dotted",
        "red",
    )
    plot = _plot_variable(
        plot,
        np_data["val_loss_ori"],
        "Val loss: orientation (rad)",
        "dotted",
        "black",
    )
    plot.legend()
    plt.grid(color="0.95")
    plt.xlabel("Epochs", **times_new_roman)
    plt.ylabel("Loss", **times_new_roman)
    plt.savefig(f"{id}.png", bbox_inches="tight", dpi=500)
    plt.show()
    # for arr in np_data:
    #     y = np_data[arr]
    #     num_epochs = range(0, len(y))

    #     print(arr)
    #     plt.plot(
    #         num_epochs,
    #         y,
    #         label=f"{arr}",
    #         linestyle=styles_dict[arr]["style"],
    #         color=styles_dict[arr]["color"],
    #     )
    # plt.title("Loss Curve")
    # plt.xlabel("Epochs", **times_new_roman)
    # plt.ylabel("Loss", **times_new_roman)
    # plt.show()


def main():
    matplotlib.rc('font', family="Times New Roman")
    for file in files_to_plot:
        plot_training_loss(files_to_plot[file], file)


if __name__ == "__main__":
    main()
