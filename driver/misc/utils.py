# Copyright (C) 2021 Weixuan Zhang, Jason Brown
#
# SPDX-License-Identifier: MIT
"""This module contains some helper functions.
"""

from functools import partial
from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np


def ensure_list_or_tuple(item):
    if isinstance(item, (list, tuple)):
        return item
    elif item is not None:
        return [item]
    else:
        return None


def flatten_iterable(x):
    result = []
    for item in x:
        if hasattr(item, "__iter__") and not isinstance(item, str):
            result.extend(flatten_iterable(item))
        else:
            result.append(item)
    return result


def fire_and_forget(function, *args, **kwargs):
    """
    Execute a function separately whilst carrying on with the rest of the programs execution.
    Function MUST be accessible from main scope i.e. is not nested in another function. It can be a method from a class.

    :param function: Function to execute
    :param args: Arguments for that function
    :param kwargs: Keyword arguments for that function
    """
    process = Process(target=function, args=args, kwargs=kwargs)
    process.start()


def get_rotation_matrix(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def rotate_vector(vec: np.ndarray, angle: float) -> np.ndarray:
    return (get_rotation_matrix(angle) @ vec.reshape(2, 1)).flatten()


def point_in_line_shadow(p1: np.ndarray, p2: np.ndarray, q: np.ndarray) -> bool:
    segment_length = np.linalg.norm(p2 - p1)
    segment_dir = (p2 - p1) / segment_length
    projection = np.dot(q - p1, segment_dir)

    return 0 < projection < segment_length


def get_min_distance_to_segment(p1: np.ndarray, p2: np.ndarray, q: np.ndarray) -> float:
    return np.linalg.norm(np.cross(
        (p2 - p1) / np.linalg.norm(p2 - p1),
        q - p1
    )) if point_in_line_shadow(p1, p2, q) else min(
        np.linalg.norm(q - p1), np.linalg.norm(q - p2)
    )


def get_rectangle_sides(vertices: list) -> list:
    return list(map(
        lambda i: (vertices[i], vertices[(i + 1) % 4]),
        range(4)
    ))


def get_min_distance_point_rectangle(rect_sides: list, q: np.ndarray) -> float:
    return min(map(
        lambda side: get_min_distance_to_segment(*side, q),
        rect_sides
    ))


def get_min_distance_rectangles(r1: list, r2: list) -> float:
    r1 = list(map(np.asarray, r1))
    r2 = list(map(np.asarray, r2))

    min_r1_to_r2 = min(map(
        partial(
            get_min_distance_point_rectangle,
            get_rectangle_sides(r2)
        ),
        r1
    ))

    min_r2_to_r1 = min(map(
        partial(
            get_min_distance_point_rectangle,
            get_rectangle_sides(r1)
        ),
        r2
    ))

    return min(min_r1_to_r2, min_r2_to_r1)


def print_if_debug(*args, debug_flag=True):
    if debug_flag:
        print(*args)


if __name__ == "__main__":
    rect1 = [np.array([0, 1]), np.array([1, 1]), np.array([1, 0]), np.array([0, 0])]
    rect2 = [np.array([2, 3]), np.array([3, 3]), np.array([3, 2]), np.array([2, 2])]
    print(get_min_distance_rectangles(rect1, rect2))
