# Copyright (C) 2021 Jason Brown
#
# SPDX-License-Identifier: MIT
"""Class file for detected objects handler"""


class ObjectDetectionHandler:
    def __init__(self):
        """Class for handling object detections"""
        self.last_object_id = 0
        self.objects = {}

    def clear_all_objects(self):
        """Removes all detections but leaves id counter alone"""
        self.objects = {}

    @property
    def num_objects(self):
        """How many objects we currently have stored"""
        return len(self.objects)

    def new_detection(self, position: list, classification: str):
        """Add a new detected object to the handler

        Args:
            position ([float, float]): Objects co-ordinates, East-North, m
            classification (string): The kind of object this is (Word 'kind' used to avoid confusion with programming type)
        """

        # Validation on classification string
        valid_classifications = ["unknown", "robot", "box", "red_box", "green_box", "wall"]
        classification = classification.lower()
        if classification not in valid_classifications:
            raise Exception(f"{classification} is an invalid classification for a detected object\n"
                            f"Valid: {', '.join(valid_classifications)}")

        self.objects[self.last_object_id] = {
            "id": self.last_object_id,
            "position": position,
            "class": classification
        }

        self.last_object_id += 1

    def remove_detection(self, identity: int):
        """Remove a detected object via its identity

        Args:
            identity (int): Identity of object to remove
        """
        del self.objects[identity]

    def get_sorted_objects(self, valid_classes: list, key=None) -> list:
        """Returns a list of object dicts based on a sorting algorithm

        Args:
            valid_classes (list): The valid classes of objects to return
            sort_quantity (string): The quantity of the object to give the algorithm
            sort_algorithm (function): The algorithm used to sort the blocks

        Returns:
            list: The object positions sorted by the algorithm
        """

        return sorted([v for v in self.objects.values() if v["class"] in valid_classes], key=key)
