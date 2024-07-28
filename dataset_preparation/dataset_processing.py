from pathlib import Path
import math
import numpy as np


# Convention: Start at front left, go counter-clockwise
CHASER_MARKERS = [5, 7, 1, 3]
TARGET_MARKERS = [13, 15, 9, 11]


def get_corners(markers, MARKER_FILTER):
    corners = []
    for marker in markers:
        if marker["id"] in MARKER_FILTER:
            corners.append(marker)
    return corners


def calculate_center(corners):
    x_total = 0
    y_total = 0
    for corner in corners:
        x = float(corner["x"]) / 1000
        y = float(corner["y"]) / 1000
        x_total = x_total + x
        y_total = y_total + y
    return (x_total / 4, y_total / 4)


def calculate_angle(markers):
    front_left = markers[0]
    back_left = markers[1]

    dx = back_left["x"] - front_left["x"]
    dy = back_left["y"] - front_left["y"]
    return math.atan2(dy, dx)


def calculate_global_pose(markers):
    x, y = calculate_center(markers)
    yaw = calculate_angle(markers)

    return x, y, yaw


def convert_to_homogenous(pose):
    x, y, yaw = pose
    c = np.cos(yaw)
    s = np.sin(yaw)

    transform = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
    return transform


def extract_rotation_matrix(T):
    return T[:2, :2]


def calculate_relative_pose(T_chaser, T_target):
    x_chaser, y_chaser = T_chaser[:2, 2]
    x_target, y_target = T_target[:2, 2]
    R_chaser = T_chaser[:2, :2]
    R_target = T_target[:2, :2]

    r_chaser = math.atan2(R_chaser[1, 0], R_chaser[0, 0])
    r_target = math.atan2(R_target[1, 0], R_target[0, 0])

    x, y = (x_target - x_chaser), (y_target - y_chaser)
    x, y = R_chaser.T @ np.array([x, y])
    theta = r_target - r_chaser
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    return x, y, theta


def get_default_filepath(directory) -> Path:
    name = f"{Path(directory).name}.json"
    path = Path(directory).joinpath(name)
    return path


def get_relative_pose_from_markers(
    markers, chaser_markers=CHASER_MARKERS, target_markers=TARGET_MARKERS
):
    pose_chaser = calculate_global_pose(get_corners(markers, chaser_markers))
    pose_target = calculate_global_pose(get_corners(markers, target_markers))

    T_chaser2world = convert_to_homogenous(pose_chaser)
    T_target2world = convert_to_homogenous(pose_target)

    return calculate_relative_pose(T_chaser2world, T_target2world)
