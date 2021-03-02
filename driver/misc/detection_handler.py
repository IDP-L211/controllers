# Copyright (C) 2021 Jason Brown, Weixuan Zhang
#
# SPDX-License-Identifier: MIT
"""Class file for detected objects handler"""

from itertools import starmap


class Target:
    """Class representing the target"""

    def __init__(self, position: list, classification: str):
        self.position = position
        self.classification = classification

    @property
    def profit(self):
        return 1

    def is_near(self, position: list, threshold: float = 0.2):
        return all(starmap(lambda rp, p: abs(rp - p) < threshold, zip(self.position, position)))

    def __repr__(self):
        return f'{self.classification} at {self.position}'

    def __lt__(self, other):
        return self.profit < other.profit

    def __eq__(self, other):
        return self.profit < other.profit


class ObjectDetectionHandler:
    """Class for handling object detections"""

    def __init__(self):
        self.objects = []

    def clear_all_objects(self):
        """Removes all detections but leaves id counter alone"""
        self.objects = []

    @property
    def num_objects(self):
        """How many objects we currently have stored"""
        return len(self.objects)

    def new_detection(self, position: list, classification: str):
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

        self.objects.append(Target(position, classification))

    def remove_detection(self, target: Target):
        """Remove a detected object via its identity

        Args:
            target: The target object to be removed
        """
        self.objects.remove(target)

    def get_sorted_objects(self, valid_classes: list, key=None) -> list:
        """Returns a list of object dicts based on a sorting algorithm

        Args:
            valid_classes (list): The valid classes of objects to return
            valid_classes (string): The quantity of the object to give the algorithm
            key (Callable): Callable that returns the key used to sort

        Returns:
            list: The object positions sorted by the algorithm
        """

        return sorted(
            filter(lambda target: target.classification in valid_classes, self.objects),
            key=key
        )
