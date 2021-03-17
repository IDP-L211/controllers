# Copyright (C) 2021 Jason Brown, Weixuan Zhang
#
# SPDX-License-Identifier: MIT
"""Class file for detected objects handler"""

from typing import Union, Callable
from collections import defaultdict
from functools import partial
from operator import attrgetter

import numpy as np
from sklearn.cluster import DBSCAN

from .geometry import get_path_rectangle, point_in_rectangle


class TargetingHandler:
    """Class for targeting, finding potential block positions from sensor data"""

    def __init__(self):
        self.positions = []
        # self.bounds = []
        self.scan_positions = []

        self.relocating = False
        self.next_scan_position = []

        self.last_fallback_scan_quadrant = None

    @property
    def num_scans(self) -> int:
        return len(self.scan_positions)

    @staticmethod
    def get_centroid(coord_list) -> np.ndarray:
        """Get the centroid of a list of coordinates

        Args:
            coord_list (list): List of coordinates, [[x1,y1], ...]

        Returns:
            np.ndarray: The centroid
        """
        return np.mean(np.asarray(coord_list), axis=0)

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
        """Get the next scan position if the last scan returns no suitable target

        The next scan position is determined by taking the perpendicular bisector, giving triangles with hypotenuse
        equalling (the maximum range of the distance sensor - 0.1)m, either side of the line segment connecting
        the latest two scanning positions. Then the coordinate of this position is clipped by a
        specified value.

        Args:
            sensor_max_range (float): The maximum range of the distance sensor
            clip (float): Maximum value of the x and y coordinates of the fallback position

        Returns:
            list: The fallback scan position
        """
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

        dist = np.sqrt((sensor_max_range - 0.1) ** 2 - diff_norm ** 2 / 4)
        a = clipped(prev_position + 0.5 * diff + np.array([-1, 1]) * np.flip(diff_unit) * dist)
        b = clipped(prev_position + 0.5 * diff + np.array([1, -1]) * np.flip(diff_unit) * dist)

        return sorted([a, b], key=lambda p: np.linalg.norm(scan_centroid - p))[1]

    def get_fallback_scan_position2(self, bot_position: list):
        # One position in each quadrant, indexed for xy cartesian coords, optimised to avoid bots at home when pathing
        scan_positions = [[[-0.7, -0.6], [-0.7, 0.6]],
                          [[0.7, -0.6], [0.7, 0.6]]]
        bot_quadrant = [int(0.5 + (0.5 * np.sign(bot_position[0]))),
                        int(0.5 + (0.5 * np.sign(bot_position[1])))]

        def rotate_quadrant_clockwise(pos):
            return [pos[1], 0 if pos[0] == 1 else 1]

        scan_quadrant = rotate_quadrant_clockwise(bot_quadrant)

        # Check we haven't already just tried to scan this quadrant
        if scan_quadrant == self.last_fallback_scan_quadrant:
            scan_quadrant = rotate_quadrant_clockwise(scan_quadrant)

        self.last_fallback_scan_quadrant = scan_quadrant
        return scan_positions[scan_quadrant[0]][scan_quadrant[1]]



class Target:
    """Class representing the target"""

    def __init__(self, position: list, classification: str):
        self.position = position
        self.classification = classification
        self.sent_to_other_bot = False  # To avoid spamming the other bot and overwriting things

    @property
    def profit(self):  # Not implemented
        return 1

    @staticmethod
    def check_near(p1: Union[list, np.ndarray], p2: Union[list, np.ndarray], threshold: float) -> bool:
        return np.allclose(np.asarray(p1), np.asarray(p2), atol=threshold)

    def is_near(self, position: list, threshold: float = 0.1) -> bool:
        """Check if the target is close to the specified position

        Args:
            position (list): Position to check against
            threshold (float): The threshold

        Returns:
            bool: Whether the target is closeby
        """
        return Target.check_near(self.position, position, threshold)

    def __repr__(self):
        return f'{self.classification} at {self.position}'

    def __lt__(self, other):  # Not implemented
        return self.profit < other.profit

    def __eq__(self, other):
        return self.is_near(other.position, 0.01)


class TargetCache:
    """Class for handling object detections"""

    def __init__(self):
        self.targets = []
        self.collected = []

    @property
    def num_targets(self) -> int:
        """How many objects we currently have stored"""
        return len(self.targets)

    @property
    def num_collected(self) -> int:
        return len(self.collected)

    @property
    def num_discard(self) -> int:
        return len(list(filter(
            lambda t: t.classification == 'discard',
            self.targets
        )))

    def clear_targets(self) -> None:
        """Removes all detections but leaves id counter alone"""
        self.targets = []

    def get_valid_target_num(self, color: str) -> int:
        return len(list(filter(
            lambda t: t.classification in ['box', f'{color}_box'],
            self.targets
        )))

    def reset_discarded(self) -> None:
        for t in self.targets:
            if t.classification == 'discard':
                t.classification = 'box'

    def update_flipped(self, color: str) -> None:
        for t in self.targets:
            if t.classification == 'flipped':
                t.classification = f'{color}_box'

    def add_target(self, position: list, classification: str = 'box', min_distance_from_wall: float = 0.08) -> None:
        """Add a new detected object to the handler

        Args:
            position ([float, float]): Objects co-ordinates, East-North, m
            classification (string): The kind of object this is
                (Word 'kind' used to avoid confusion with programming type)
            min_distance_from_wall (float): Minimum distance between the target and the wall, if less than the minimum.
                the target will be discarded
        """

        # Validation on classification string
        valid_classifications = ["robot", "box", "red_box", "green_box", "discard"]
        classification = classification.lower()
        if classification not in valid_classifications:
            raise Exception(f"{classification} is an invalid classification for a detected object\n"
                            f"Valid: {', '.join(valid_classifications)}")

        if any(map(lambda x: abs(x) > 1.2 - min_distance_from_wall, position)):  # target too close to the wall
            classification = 'discard'

        for t in self.targets:  # check if the same target already exist in cache
            if t.is_near(position):
                if t.classification == 'robot' and classification != 'robot':  # updates if it was classified as robot
                    # unlikely the other robot is at the same position again, probably false classification last time
                    t.classification = classification
                elif classification in ['red_box', 'green_box', 'discard', 'flipped']:
                    # this is when the other robot sends in the confirmed colour and position
                    t.classification = classification
                t.position = TargetingHandler.get_centroid([t.position, position])  # more accurate position
                break
            if t.classification == 'robot' and classification == 'robot':
                # there should only be one position of the other robot
                t.position = position
                break
        else:
            self.targets.append(Target(position, classification))

    def remove_target(self, target: Target) -> None:
        """Remove a detected object via its identity

        Args:
            target: The target object to be removed
        """
        try:
            self.targets.remove(target)
        except ValueError:
            print(f'[WARNING] {target} not in cache')

    def get_targets(self, valid_classes: list, key: Callable = None) -> list:
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

    def pop_target(self, valid_classes, key: Callable = None) -> Union[Target, None]:
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

    def check_target_path_blocked(self, curr_target_position: list, curr_position: list,
                                  other_bot_pos: Union[list, None] = None,
                                  other_bot_vertices: Union[list, None] = None) -> bool:
        """Check if other targets or the other robot is in the path to currently selected target

        Args:
            curr_target_position (list): The target chosen
            curr_position (list): Current position of the centre of the robot
            other_bot_pos (list, None): Current position of the other robot, optional
            other_bot_vertices (list, None): Current positions of vertices of the other robot, optional


        Returns:
            bool: Whether the path is blocked
        """
        check_in_path = partial(
            point_in_rectangle,
            get_path_rectangle(np.asarray(curr_target_position), np.asarray(curr_position))
        )

        return any(map(
            check_in_path,
            map(
                attrgetter('position'),
                filter(
                    # TODO this can be potentially changed to only checking blocks of the wrong colour
                    lambda t: not t.is_near(curr_target_position, 0.01) \
                              and t.classification in ['box', 'red_box', 'green_box', 'flipped'],
                    self.targets
                )
            )
        )) or (check_in_path(other_bot_pos) if other_bot_pos else False) or (any(map(
            check_in_path,
            other_bot_vertices
        )) if other_bot_vertices else False)

    def prepare_collected_message(self) -> list:
        """Prepare a list of positions where blocks were collected

        This is used to prepare the message to the other robot.

        Returns:
            list: List of coordinates (as lists instead of np.ndarray)
        """
        return list(map(
            list,
            map(attrgetter('position'), self.collected)
        ))

    def remove_collected_by_other(self, collected: Union[list, None]) -> None:
        """Remove the targets already collected by the other robot

        Args:
            collected (list, None): List of coordinates where the blocks were collected
        """
        if collected is None:
            return

        for collected_pos in collected:
            for t in self.targets:
                if t.is_near(collected_pos):
                    self.remove_target(t)
