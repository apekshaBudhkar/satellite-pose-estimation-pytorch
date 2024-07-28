import matplotlib.pyplot as plt
import matplotlib.colors as clr

import json
from pathlib import Path
import click
import numpy as np

from matplotlib import cm


def get_json_data(file_path):
    with open(Path(file_path), "r") as file:
        return json.load(file)


def create_plot(x, y, yaw):
    times_new_roman = {"fontname": "Times New Roman", "size": 12}

    ax0 = plt.subplot(1, 3, 1)
    plt.hist(x, density=False, color="#a20025", edgecolor="white")
    plt.xlabel("X distances (m)", **times_new_roman)
    plt.ylabel("Distribution (count)", **times_new_roman)


    ax1 = plt.subplot(1, 3, 2)
    plt.hist(y, density=False, color="#a20025", edgecolor="white")
    plt.xlabel("Y distances (m)", **times_new_roman)
    ax1.get_yaxis().set_visible(False)

    ax2 = plt.subplot(1, 3, 3)
    plt.hist(yaw, density=False, color="#a20025", edgecolor="white")
    plt.xlabel("Yaw angles (radians)", **times_new_roman)
    ax2.get_yaxis().set_visible(False)

    upperbound = max(ax0.get_ylim()[1], ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax0.set_ylim(0, upperbound)
    ax1.set_ylim(0, upperbound)
    ax2.set_ylim(0, upperbound)

    times_new_roman_small = {"fontname": "Times New Roman", "size": 10}

    ax0.set_xticklabels(ax0.get_xticks(), **times_new_roman)
    ax1.set_xticklabels(ax1.get_xticks(), **times_new_roman)
    ax2.set_xticklabels(ax2.get_xticks(), **times_new_roman)

    ax0.set_yticklabels(ax0.get_yticks(), **times_new_roman)




def plot_3d():
    t = np.arange(len(data))

    ax_3d = plt.figure().add_subplot(projection="3d")

    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("yaw")
    # cmap = cm.get_cmap("inferno_r")
    norm = plt.Normalize(vmin=-3, vmax=3)
    n_c = clr.CenteredNorm(-3, 3)
    print(norm)
    ax_3d.scatter(x, y, yaw, s=2.5)


@click.command()
@click.option("-i", "--input_directory", required=True)
def main(input_directory):
    splits = ["train", "test", "validation"]
    for split in splits:
        data = get_json_data(f"{input_directory}/{split}/{split}.json")
        print(split)
        print(f" Original size {len(data)}")
        x = []
        y = []
        yaw = []

        for key in data:
            pt = data[key]
            dx = pt["x"]
            dy = pt["y"]
            dyaw = pt["yaw"]
            x.append(dx)
            y.append(dy)
            yaw.append(dyaw)
        print(len(data))

        create_plot(x, y, yaw)
        plt.savefig(f"{split}.png", dpi=500)
        plt.close()


if __name__ == "__main__":
    main()
