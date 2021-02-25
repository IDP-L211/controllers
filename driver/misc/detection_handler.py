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

    def get_box_positions_list(self, robot) -> list:
        """Returns a list of box positions sorted by distance, smallest first

        Args:
            robot: The robot which this instance is attached to
                This is to use it's distance_from_bot method for the sorting

        Returns:
            list: The object positions sorted by distance from bot"""
        return sorted([v["position"] for v in self.objects.values() if v["class"] in ["box", "red_box", "green_box"]],
                      key=robot.distance_from_bot)
