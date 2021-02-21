# Copyright (C) 2021 Weixuan Zhang
#
# SPDX-License-Identifier: MIT
"""This module contains some helper functions.
"""

import numpy as np


def get_rotation_matrix(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def rotate_vector(vec: np.ndarray, angle: float) -> np.ndarray:
    return (get_rotation_matrix(angle) @ vec.reshape(2, 1)).flatten()


def get_target_bearing(own_pos: list, target_pos: list) -> float:
    """
    Gets the bearing from North to a target position, based on our own position

    Args:
        own_pos: [float, float]: Position of current position (East, North) in meters
        target_pos: [float, float]: Position of target position (East, North) in meters
    """

    if own_pos == target_pos:
        raise Exception("Positions are the same, bearing is undefined")

    # Convert lists to np arrays
    own_pos = np.array(own_pos)
    target_pos = np.array(target_pos)

    # Get our direction vectors
    target_vector = target_pos - own_pos
    target_vector_direction = target_vector / np.linalg.norm(target_vector)
    north = np.array([0, 1])

    # Get angle and correct if our target is westerly
    angle = np.arccos(np.clip(np.dot(north, target_vector_direction), -1.0, 1.0))
    bearing = angle if target_vector[0] >= 0 else 2 * np.pi - angle

    return bearing
