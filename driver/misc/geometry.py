# Copyright (C) 2021 Weixuan Zhang
#
# SPDX-License-Identifier: MIT
"""This module contains functions related to geometry calculations
"""

from functools import partial

import numpy as np


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


def get_path_rectangle(target_position: np.ndarray, curr_position: np.ndarray) -> list:
    diff = target_position - curr_position
    diff_norm = np.linalg.norm(diff)
    diff_unit = diff / diff_norm

    target_to_topleft = rotate_vector(diff_unit, np.pi / 2)

    # using magic numbers, forgive me...
    # these numbers corresponds to the displacement of the font vertices of the robot bounding box
    # from the robot centre
    topleft = target_position + target_to_topleft * 0.15
    topright = target_position - target_to_topleft * 0.225

    bottomleft = topleft - diff_unit * (diff_norm - 0.075)
    bottomright = topright - diff_unit * (diff_norm - 0.075)

    return [topleft, topright, bottomright, bottomleft]


def point_in_rectangle(rect_vertices: list, p: np.ndarray) -> bool:
    sides = get_rectangle_sides(rect_vertices)
    return all(map(
        lambda s: point_in_line_shadow(*s, p),
        sides
    ))


if __name__ == "__main__":
    rect1 = [np.array([0, 1]), np.array([1, 1]), np.array([1, 0]), np.array([0, 0])]
    rect2 = [np.array([2, 3]), np.array([3, 3]), np.array([3, 2]), np.array([2, 2])]
    print(get_min_distance_rectangles(rect1, rect2))

    print(get_path_rectangle(np.array([0, 1]), np.array([0, 0])))

    print(point_in_rectangle(rect1, np.array([0.5, 0.5])))
    print(point_in_rectangle(rect1, np.array([2, 0.5])))
