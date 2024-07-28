import numpy as np
from pathlib import Path
import json
import math

from matplotlib import pyplot as plt

from ground_truth_processing import get_marker_metadata
from dataset_processing import calculate_global_pose, get_corners

TEST_SET_PATH = "/media/apeksha/T7/spot_data_2024/2024/experimentv5/test/raw"
MODEL_OUTPUT_PATH = "/media/apeksha/T7/satellite-pose-estimation-infrared/outputs/apr7/epoch30_yolo10_2024-4-7_1-18.json"

TIME_PER_IMAGE = 0.01910655981
# Convention: Start at front left, go counter-clockwise
CHASER_MARKERS = [5, 7, 1, 3]
TARGET_MARKERS = [13, 15, 9, 11]

times_new_roman = {"fontname": "Times New Roman", "size": 16}


def OrientationLoss(prediction, actual):
    prediction = (prediction + math.pi) % (2 * math.pi) - math.pi
    actual = (actual + math.pi) % (2 * math.pi) - math.pi

    angle_diff = prediction - actual
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
    angle_error = abs(angle_diff)
    return angle_error


def get_json_data_from_file(filepath: str) -> json:
    with open(filepath, "r") as data:
        return json.load(data)


def sort_data(model_outputs):
    data = sorted(model_outputs, key=lambda x: x["id"], reverse=False)
    return data


def create_global_pose_plot(x_chaser, y_chaser, x_target, y_target, pred_x, pred_y, ax):
    ax.scatter(x_chaser, y_chaser, marker=".", c="red", label="chaser actual position")
    ax.scatter(
        x_target, y_target, marker=".", c="black", label="target actual position"
    )
    ax.scatter(
        pred_x, pred_y, marker="x", c="green", label="target determined position"
    )

    ax.legend()
    ax.grid()
    ax.set_title("Determinations in the global frame", **times_new_roman)
    ax.set_xlim(0, 3.55)  # Global x
    ax.set_ylim(0, 3.25)  # Global y
    ax.set_xlabel("x position (m)", **times_new_roman)
    ax.set_ylabel("y position (m)", **times_new_roman)


def create_x_y_plot(x_pred, y_pred, x_actual, y_actual, ax):
    ax.scatter(x_pred, y_pred, marker=".", c="black", label="predicted")
    ax.scatter(x_actual, y_actual, marker="x", c="red", label="actual")

    ax.legend()
    ax.grid()
    ax.set_title("Predictions in relative frame (x, y)", **times_new_roman)
    ax.set_xlim(0.5, 3.0)  # Relative x
    ax.set_ylim(-1.5, 1.5)  # Relative y
    ax.set_xlabel("Relative x position (m)")
    ax.set_ylabel("Relative y position (m)")


def create_x_yaw_plot(x_pred, yaw_pred, x_actual, yaw_actual, ax):
    ax.scatter(x_pred, yaw_pred, marker=".", c="black", label="predicted")
    ax.scatter(x_actual, yaw_actual, marker="x", c="red", label="actual")

    ax.legend()
    ax.grid()
    ax.set_title("Predictions in relative frame (x, yaw)", **times_new_roman)
    ax.set_xlim(1.0, 3.0)  # Relative x
    ax.set_ylim(-3.14, 3.14)  # Reltative yaw
    ax.set_xlabel("Relative x position (m)")
    ax.set_ylabel("Relative yaw angle (rad)")


def create_y_yaw_plot(y_pred, yaw_pred, y_actual, yaw_actual, ax):
    ax.scatter(y_pred, yaw_pred, marker=".", c="black", label="predicted")
    ax.scatter(y_actual, yaw_actual, marker="x", c="red", label="actual")

    ax.legend()
    ax.grid()
    ax.set_title("Predictions in relative frame (y, yaw)", **times_new_roman)
    ax.set_xlim(-1.5, 1.5)  # Relative y
    ax.set_ylim(-3.14, 3.14)  # Relative yaw
    ax.set_xlabel("Relative y position (m)")
    ax.set_ylabel("Relative yaw angle (rad)")


def create_x_vs_time_plot(x_determined, x_actual, ax, timestamps):
    first_timestamp = int(timestamps[0])
    list_original_time = []
    list_inference_times = []

    for ts in timestamps:
        original_time = int(ts) - first_timestamp
        list_original_time.append(original_time / 1000)
        list_inference_times.append((original_time + 19.1) / 1000)


    ax.scatter(list_original_time, x_actual, marker=".", c="black", label="actual")
    ax.scatter(list_inference_times, x_determined, marker="x", c="green", label="determined")

    ax.legend()
    ax.grid()

    ax.set_title("Relative x vs time", **times_new_roman)
    ax.set_ylim(0.5, 3.25)  # x
    ax.set_xlabel("Time elapsed (s)", **times_new_roman)
    ax.set_ylabel("Relative x values (m)", **times_new_roman)


def create_y_vs_time_plot(y_determined, y_actual, ax, timestamps):
    first_timestamp = int(timestamps[0])
    list_original_time = []
    list_inference_times = []

    for ts in timestamps:
        original_time = int(ts) - first_timestamp
        list_original_time.append(original_time / 1000)
        list_inference_times.append((original_time + 19.1) / 1000)

    ax.scatter(list_original_time, y_actual, marker=".", c="black", label="actual")
    ax.scatter(list_inference_times, y_determined, marker="x", c="green", label="determined")

    ax.legend()
    ax.grid()

    ax.set_title("Relative y vs time", **times_new_roman)
    ax.set_ylim(-1.0, 1.0)  # x
    ax.set_xlabel("Time elapsed (s)", **times_new_roman)
    ax.set_ylabel("Relative y values (m)", **times_new_roman)


def create_yaw_vs_time_plot(yaw_determined, yaw_actual, ax, timestamps):
    first_timestamp = int(timestamps[0])
    list_original_time = []
    list_inference_times = []

    for ts in timestamps:
        original_time = int(ts) - first_timestamp
        list_original_time.append(original_time / 1000)
        list_inference_times.append((original_time + 19.1) / 1000)

    ax.scatter(list_original_time, yaw_actual, marker=".", c="black", label="actual")
    ax.scatter(list_inference_times, yaw_determined, marker="x", c="green", label="determined")

    ax.legend()
    ax.grid()

    ax.set_title("Relative yaw vs time", **times_new_roman)
    ax.set_ylim(-3.14, 3.14)  # x
    ax.set_xlabel("Time elapsed (s)", **times_new_roman)
    ax.set_ylabel("Relative yaw values (rad)", **times_new_roman)


def get_2d_rotation_mat(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    mat = np.array([[c, -s], [s, c]])
    return mat


def plot_all(json_data, fname):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    list_chaser_global_x = []
    list_chaser_global_y = []
    list_target_global_x = []
    list_target_global_y = []

    list_predicted_pose_x_in_global_frame = []
    list_predicted_pose_y_in_global_frame = []

    list_predicted_x = []
    list_predicted_y = []
    list_predicted_yaw = []

    list_actual_x = []
    list_actual_y = []
    list_actual_yaw = []

    list_loss_x = []
    list_loss_y = []
    list_loss_yaw = []

    list_timestamps = []
    for data_point in json_data:
        id: str = data_point["id"][0]
        timestamp = id[id.find("_") + 1 : id.find(".png")]
        list_timestamps.append(timestamp)

        txt = Path(id).with_suffix(".txt")
        raw_marker_path = Path(TEST_SET_PATH).joinpath(txt)

        markers = get_marker_metadata(str(raw_marker_path))
        pose_chaser = calculate_global_pose(get_corners(markers, CHASER_MARKERS))
        pose_target = calculate_global_pose(get_corners(markers, TARGET_MARKERS))

        list_chaser_global_x.append(pose_chaser[0])
        list_chaser_global_y.append(pose_chaser[1])
        list_target_global_x.append(pose_target[0])
        list_target_global_y.append(pose_target[1])

        rotation_mat_chaser = get_2d_rotation_mat(pose_chaser[2])

        actual_x, actual_y, actual_yaw = data_point["actual"]
        prediction_x, prediction_y, prediction_yaw = data_point["predictions"]

        result = rotation_mat_chaser @ [prediction_x, prediction_y]
        calculated_x = pose_chaser[0] + result[0]
        calculated_y = pose_chaser[1] + result[1]

        list_predicted_pose_x_in_global_frame.append(calculated_x)
        list_predicted_pose_y_in_global_frame.append(calculated_y)

        list_predicted_x.append(prediction_x)
        list_predicted_y.append(prediction_y)
        list_predicted_yaw.append(prediction_yaw)

        list_actual_x.append(actual_x)
        list_actual_y.append(actual_y)
        list_actual_yaw.append(actual_yaw)

        x_loss, y_loss, yaw_loss = (
            abs(actual_x - prediction_x),
            abs(actual_y - prediction_y),
            OrientationLoss(prediction_yaw, actual_yaw),
        )
        list_loss_x.append(x_loss)
        list_loss_y.append(y_loss)
        list_loss_yaw.append(yaw_loss)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Plot 1: Global pose
    create_global_pose_plot(
        list_chaser_global_x,
        list_chaser_global_y,
        list_target_global_x,
        list_target_global_y,
        list_predicted_pose_x_in_global_frame,
        list_predicted_pose_y_in_global_frame,
        axes[0, 0],
    )

    # Plot 2, 3, and 4
    create_x_vs_time_plot(list_predicted_x, list_actual_x, axes[0, 1], list_timestamps)
    create_y_vs_time_plot(list_predicted_y, list_actual_y, axes[1, 0], list_timestamps)
    create_yaw_vs_time_plot(list_predicted_yaw, list_actual_yaw, axes[1, 1], list_timestamps)

    f = open(f"outputs_timed/{fname}.txt", "w")
    f.write(f"mean_x_loss = {np.mean(list_loss_x)}\n")
    f.write(f"mean_y_loss = {np.mean(list_loss_y)}\n")
    f.write(f"mean_yaw_loss = {np.mean(list_loss_yaw)}\n")
    f.write(f"med_x_loss = {np.median(list_loss_x)}\n")
    f.write(f"med_y_loss = {np.median(list_loss_y)}\n")
    f.write(f"med_yaw_loss = {np.median(list_loss_yaw)}\n")
    f.close()

    fig.savefig(f"outputs_timed/{fname}.png", bbox_inches="tight", dpi=500)
    plt.close()


"""
List of scenarios:


 #    Range       Desc. 
 1 |  54-126, 198-216, 1008-1026             |  change x                                  
 2 |  0-20, 1152-1188, 1494-1511             |  change y               
 3 |  234-260 (approximate), 1980-1997       |  change yaw             
 4 |  1764-1817, 2196-2213, 2466-2500        |  change x, and y                
 5 |  144-180, 306-342, 432-504, 1368-1385, 1710-1745, 2232-2265, 2412-   |  change y, and yaw              
 6 |  1278-1296, 1332-1367, 1548-1600        |  change x, and yaw              
 7 |  288-306, 1872-1925                     |  change x, y, and yaw             
 8 |  1944-1961                              |  change x, y, and yaw, while chaser moves                 
 9 |  882-900,918-936                        |  change x, while chaser moves
 10 | 2322-2340                              |  chaser and target move together while yaw is changing
11 | 1926 - 1962
"""


# def main():
#     json_data = get_json_data_from_file(MODEL_OUTPUT_PATH)["model_performance"]
#     json_data = sort_data(json_data)
#     step_size = 18
#     for i in range(0, len(json_data), step_size):
#         plot_all(json_data[i : i + step_size], i)


def scenario_1():
    json_data = get_json_data_from_file(MODEL_OUTPUT_PATH)["model_performance"]
    json_data = sort_data(json_data)

    x_range = json_data[63:105]
    plot_all(x_range, "scenario1_1")

    x_range = json_data[200:221]
    plot_all(x_range, "scenario1_2")


def scenario_2():
    json_data = get_json_data_from_file(MODEL_OUTPUT_PATH)["model_performance"]
    json_data = sort_data(json_data)

    range = json_data[0:20]
    plot_all(range, "scenario2_1")

    range = json_data[1159:1178]
    plot_all(range, "scenario2_2")


def scenario_3():
    json_data = get_json_data_from_file(MODEL_OUTPUT_PATH)["model_performance"]
    json_data = sort_data(json_data)

    range = json_data[237:250]
    plot_all(range, "scenario3_1")
    range = json_data[1980:1997]
    plot_all(range, "scenario3_2")


def scenario_4():
    json_data = get_json_data_from_file(MODEL_OUTPUT_PATH)["model_performance"]
    json_data = sort_data(json_data)

    range = json_data[1776:1793]
    plot_all(range, "scenario4_1")
    range = json_data[2194:2215]
    plot_all(range, "scenario4_2")


def scenario_5():
    json_data = get_json_data_from_file(MODEL_OUTPUT_PATH)["model_performance"]
    json_data = sort_data(json_data)

    range = json_data[1764:1817]
    plot_all(range, "scenario5_1")
    range = json_data[144:180]
    plot_all(range, "scenario5_2")


def scenario_6():
    json_data = get_json_data_from_file(MODEL_OUTPUT_PATH)["model_performance"]
    json_data = sort_data(json_data)

    range = json_data[1764:1817]
    plot_all(range, "scenario6_1")
    range = json_data[1332:1367]
    plot_all(range, "scenario6_2")


def scenario_7():
    json_data = get_json_data_from_file(MODEL_OUTPUT_PATH)["model_performance"]
    json_data = sort_data(json_data)

    range = json_data[1764:1817]
    plot_all(range, "scenario7_1")
    range = json_data[288:306]
    plot_all(range, "scenario7_2")


def scenario_8():
    json_data = get_json_data_from_file(MODEL_OUTPUT_PATH)["model_performance"]
    json_data = sort_data(json_data)

    range = json_data[1764:1817]
    plot_all(range, "scenario8_1")
    range = json_data[1926:1962]
    plot_all(range, "scenario8_2")
    range = json_data[2001:2021]
    plot_all(range, "scenario8_3")


def main():
    scenario_1()
    scenario_2()
    scenario_3()
    scenario_4()
    scenario_5()
    scenario_6()
    scenario_7()
    scenario_8()


if __name__ == "__main__":
    main()
