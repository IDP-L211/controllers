# Copyright (C) 2021 Jason Brown, Weixuan Zhang
#
# SPDX-License-Identifier: MIT
"""Class file for detected objects handler"""

from typing import Union
from itertools import starmap
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN


class TargetingHandler:
    """Class for targeting, finding potential block positions from sensor data"""

    def __init__(self):
        self.positions = []
        # self.bounds = []
        self.scan_positions = []

        self.relocating = False
        self.next_scan_position = []

    @property
    def num_scans(self) -> int:
        return len(self.scan_positions)

    @staticmethod
    def get_centroid(coord_list) -> np.ndarray:
        return np.array([sum(c) / len(c) for c in zip(*coord_list)])

    def clear_cache(self) -> None:
        self.positions = []
        # self.bounds = []

    def get_targets(self, curr_position: list) -> list:
        """Calculate potential target positions from a list of position data points.

        DBSCAN clustering algorithms is used, and the average positions of points within
        a cluster is taken to be the target position.

        Returns:
            [[x,y],...]: List of target positions
        """
        self.scan_positions.append(np.asarray(curr_position))
        if len(self.positions) == 0:  # no data, prevent DBSCAN from giving error
            return []

        targets = defaultdict(list)
        labels = DBSCAN(eps=0.05, min_samples=3).fit(self.positions).labels_
        for i, label in enumerate(labels):
            if label == -1:
                continue
            targets[label].append(self.positions[i])

        self.clear_cache()

        return list(map(
            TargetingHandler.get_centroid,
            targets.values()
        ))

    def get_fallback_scan_position(self, sensor_max_range: float, clip: float = 0.6) -> list:
        if self.num_scans < 1:
            raise RuntimeError('No scan completed, do a scan first')

        def clipped(pos):
            return list(map(
                lambda x: min(max(-clip, x), clip),
                pos
            ))

        curr_position = self.scan_positions[-1]
        # very unlikely to get no targets on the first two scans
        if self.num_scans < 3:
            return clipped(curr_position - sensor_max_range * curr_position / np.linalg.norm(curr_position))

        scan_centroid = TargetingHandler.get_centroid(self.scan_positions)

        prev_position = self.scan_positions[-2]
        diff = curr_position - prev_position
        diff_norm = np.linalg.norm(diff)
        diff_unit = diff / diff_norm

        dist = np.sqrt(sensor_max_range ** 2 - diff_norm ** 2 / 4)
        a = clipped(prev_position + 0.5 * diff + np.array([-1, 1]) * np.flip(diff_unit) * dist)
        b = clipped(prev_position + 0.5 * diff + np.array([1, -1]) * np.flip(diff_unit) * dist)

        return sorted([a, b], key=lambda p: np.linalg.norm(scan_centroid - p))[1]


class Target:
    """Class representing the target"""

    def __init__(self, position: list, classification: str):
        self.position = position
        self.classification = classification

    @property
    def profit(self):  # Not implemented
        return 1

    def is_near(self, position: list, threshold: float = 0.1):
        return all(starmap(lambda rp, p: abs(rp - p) < threshold, zip(self.position, position)))

    def __repr__(self):
        return f'{self.classification} at {self.position}'

    def __lt__(self, other):  # Not implemented
        return self.profit < other.profit

    def __eq__(self, other):
        return self.is_near(other.position, 0)


class TargetCache:
    """Class for handling object detections"""

    def __init__(self):
        self.targets = []

    def clear_targets(self):
        """Removes all detections but leaves id counter alone"""
        self.targets = []

    @property
    def num_targets(self):
        """How many objects we currently have stored"""
        return len(self.targets)

    def add_target(self, position: list, classification: str = 'box'):
        """Add a new detected object to the handler

        Args:
            position ([float, float]): Objects co-ordinates, East-North, m
            classification (string): The kind of object this is
                (Word 'kind' used to avoid confusion with programming type)
        """

        # Validation on classification string
        valid_classifications = ["unknown", "robot", "box", "red_box", "green_box", "wall"]
        classification = classification.lower()
        if classification not in valid_classifications:
            raise Exception(f"{classification} is an invalid classification for a detected object\n"
                            f"Valid: {', '.join(valid_classifications)}")

        for t in self.targets:  # check if the same target already exist in cache
            if t.is_near(position):
                t.classification = classification  # updates its classification
                break
        else:
            self.targets.append(Target(position, classification))

    def remove_target(self, target: Target):
        """Remove a detected object via its identity

        Args:
            target: The target object to be removed
        """
        self.targets.remove(target)

    def get_targets(self, valid_classes: list, key=None) -> list:
        """Returns a list of object dicts based on a sorting algorithm

        Args:
            valid_classes (list): The valid classes of objects to return
            key (Callable): Callable that returns the key used to sort

        Returns:
            list: The object positions sorted by the algorithm
        """

        return sorted(
            filter(lambda target: target.classification in valid_classes, self.targets),
            key=key
        )

    def pop_target(self, valid_classes: list, key=None) -> Union[Target, None]:
        """Pops the best target

        Similar to get_targets, the targets are sorted, but only the best one is returned.
        The returned target is also popped off the target list.

        Args:
            valid_classes (list): The valid classes of objects to return
            key (Callable): Callable that returns the key used to sort

        Returns:
            list: The object positions sorted by the algorithm
        """
        if len(self.targets) == 0:
            return None

        popped = self.get_targets(valid_classes, key)[0]
        self.remove_target(popped)

        return popped
