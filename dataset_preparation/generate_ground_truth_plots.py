from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

from image_processing import process_image, is_binary_datafile
from ground_truth_processing import get_marker_metadata
from dataset_processing import get_relative_pose_from_markers

import gc
import cv2
import numpy as np
import click

"""
Generates a 2D plot showing the poses of the 2 satellites on the satellite testbed, 
along with the .png image for visual validation of the poses. Saves the results in a new directory
on disk inside the original data root.
"""

"""
Matplotlib's GUI functionality has a memory leak due to which saving a large number of figures uses up all the RAM.
The following line disables the GUI backend, which is useful when you're trying to save 1000+ of images to disk)
"""
matplotlib.use("Agg")

# Convention: Start at front left, go counter-clockwise
CHASER_MARKERS = [5, 7, 1, 3]
TARGET_MARKERS = [13, 15, 9, 11]
DATA_DIRECTORY = "/vivado/spot_data/2024/mar23_1959"


def get_arrow_plot(image, ground_truth, title):
    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    x, y, yaw = ground_truth["x"], ground_truth["y"], ground_truth["yaw"]

    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(image, cmap="gray")
    ax0.axis("off")
    ax0.set_title(f"{title}\n{x, y, np.degrees(yaw)}", fontsize=10)

    ax = plt.subplot(gs[1, 0])
    ax.set_xlim(0, 4)
    ax.set_ylim(-3, 3)

    ax.arrow(
        x,
        y,
        np.cos(yaw),
        np.sin(yaw),
        length_includes_head=True,
        head_width=0.08,
        head_length=0.2,
    )
    return fig


def get_data_plot(image, markers, title):
    pose = get_relative_pose_from_markers(markers, CHASER_MARKERS, TARGET_MARKERS)
    relative_x, relative_y, relative_yaw = pose

    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    ax0 = plt.subplot(gs[0, :])
    ax0.imshow(image, cmap="gray")
    ax0.axis("off")
    ax0.set_title(
        f"{title}\n{relative_x, relative_y, np.degrees(relative_yaw)}", fontsize=10
    )

    ax1 = plt.subplot(gs[1, 0])
    ax1.set_xlim(0, 3.5)
    ax1.set_ylim(0, 3.5)

    chaser_pts = list([m for m in markers if m["id"] in CHASER_MARKERS])
    target_pts = list([m for m in markers if m["id"] in TARGET_MARKERS])

    for marker in chaser_pts:
        x = marker["x"] / 1000
        y = marker["y"] / 1000
        ax1.plot(x, y, marker=".", color="red", linestyle="-", linewidth="1.5")
        ax1.annotate(marker["id"], (x, y))

    for marker in target_pts:
        x = marker["x"] / 1000
        y = marker["y"] / 1000
        ax1.scatter(x, y, marker=".", color="black")
        ax1.annotate(marker["id"], (x, y))

    ax2 = plt.subplot(gs[1, 1])
    ax2.set_xlim(0, 4)
    ax2.set_ylim(-3, 3)

    ax2.arrow(
        relative_x,
        relative_y,
        np.cos(pose[2]),
        np.sin(pose[2]),
        length_includes_head=True,
        head_width=0.08,
        head_length=0.2,
    )
    return fig


@click.command()
@click.option("-d", "--data_root", required=False, default=DATA_DIRECTORY)
def main(data_root):
    data_files = list(Path(data_root).iterdir())
    output_folder = Path.joinpath(Path(data_root), "ground_truth_plots")
    Path.mkdir(output_folder, exist_ok=True)
    list.sort(data_files, reverse=True)
    for file in data_files:
        if is_binary_datafile(file):
            image = process_image(str(file))
            # print("Saving ", f"{data_root}/{file.stem}.png")
            cv2.imwrite(f"{data_root}/{file.stem}.png", image)
            metadata_filename = data_root + "/" + file.stem + ".txt"
            try:
                markers = get_marker_metadata(metadata_filename)
            except:
                print(f"An error occured processing {file.stem}.txt. Skipping.")
                continue
            plot = get_data_plot(image, markers, file.stem)
            plot.savefig(f"{output_folder}/{file.stem}.png", bbox_inches="tight")

            plt.close("all")
            gc.collect()


if __name__ == "__main__":
    main()
