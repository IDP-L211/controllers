# Copyright (C) 2021 Weixuan Zhang
#
# SPDX-License-Identifier: MIT
"""This module contains some helper functions.
"""

import numpy as np


def get_rotation_matrix(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def rotate_vector(vec: np.ndarray, angle: float) -> np.ndarray:
    return (get_rotation_matrix(angle) @ vec.reshape(2, 1)).reshape(1, 2)
