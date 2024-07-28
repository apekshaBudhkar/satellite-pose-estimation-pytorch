from matplotlib import pyplot as plt
import matplotlib
from pathlib import Path
import os
import datetime
import numpy as np


times_new_roman = {"fontname": "Times New Roman", "size": 14}


directory = (
    "/media/apeksha/T7/satellite-pose-estimation-infrared/scripts/checkpoints/apr7/"
)

subdirs = {
    "resnet": "resnet_2024-4-7_2-43",
    "yolo10": "yolo10_2024-4-7_0-48",
    "yolo7": "yolo7_2024-4-7_2-6",
    "yolo5": "yolo5_2024-4-7_1-29",
}

training_times = {
    "yolo10": [],
    "yolo7": [],
    "yolo5": [],
    "resnet": [],
}

colors = {
    "yolo10": matplotlib.colors.to_hex((0.541, 0.004, 0.004)),
    "yolo7": matplotlib.colors.to_hex((0.522, 0.486, 0.471)),
    "yolo5": matplotlib.colors.to_hex((0.988, 0.102, 0.02)),
    "resnet": matplotlib.colors.to_hex((0.016, 0.031, 0.059))
}


def get_epoch_from_file_name(fname):
    return (fname.split("_")[0]).split("epoch")[1]


def main():
    epochs = [0, 5, 10, 15, 20, 25, 30, 35]
    for entry in subdirs:
        subdir = subdirs[entry]
        checkpoint_directory = f"{directory}/{subdir}"
        file_list = os.listdir(checkpoint_directory)
        file_list = [file for file in file_list if Path(file).suffix == ".pth"]

        start_time = 0

        training_times[entry].append(start_time)
        for i, file in enumerate(file_list):

            full_path = Path(directory).joinpath(subdir).joinpath(file)
            modified_time = full_path.stat().st_mtime
            if i == 0:
                start_time = modified_time
            delta = modified_time - start_time
            training_times[entry].append((delta / 60))

        training_times[entry].pop(0)
        plt.plot(epochs, training_times[entry], linestyle="dashed", color=colors[entry])
    plt.grid("both")
    plt.xlabel("Epochs completed", **times_new_roman)
    plt.ylabel("Time Elapsed (minutes)", **times_new_roman)

    plt.legend(subdirs.keys())
    plt.savefig("time.png", bbox_inches="tight", dpi=500)
    plt.show()


if __name__ == "__main__":
    matplotlib.rc('font', family="Times New Roman")
    main()
