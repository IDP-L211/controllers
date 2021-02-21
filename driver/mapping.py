# Copyright (C) 2021 Weixuan Zhang
#
# SPDX-License-Identifier: MIT
"""This module contains classes for displaying a real time map of the arena
from sensor data.
"""

import numpy as np

from controller import Display


class Map(Display):
    def __init__(self, robot, arena_length: float, name: str = 'map'):
        super().__init__(name)
        self.robot = robot
        self.arena_length = arena_length
        self.width = self.getWidth()
        self.height = self.getHeight()
        self.map_length = min(self.width, self.height)

    @staticmethod
    def coord_to_xylists(list_of_arrays: list) -> list:
        """Separate the x and y coordinates into two lists

        Args:
            list_of_arrays: [[x1, y1], [x2, y2]...]

        Returns:
            list: [[x1, x2, x3...], [y1, y2, y3...]]
        """
        return np.vstack(list_of_arrays).T.tolist()

    @staticmethod
    def convert_unit(vec: np.ndarray, arena_length: float, map_length: float) -> np.ndarray:
        """Convert a meters to pixels
        """
        return vec / arena_length * map_length

    @staticmethod
    def coordtransform_world_to_map(vec: np.ndarray, arena_length: float, map_length: float) -> np.ndarray:
        """Transfer the world coordinate (East, North) to image coordinate

        The image coordinate has (0,0) at the top left corner and (width-1,height-1) at
        the bottom right corner. Therefore, the North coordinate needs to be reversed.
        The origin also need to be shifted to the center.
        """
        return (Map.convert_unit(vec, arena_length, map_length) * np.array([1, -1])
                + np.array([map_length] * 2) / 2).astype(int)

    def get_map_bot_vertices(self) -> list:
        vertices = self.robot.get_bot_vertices()
        vertices_trans = [Map.coordtransform_world_to_map(v, self.arena_length, self.map_length) for v in vertices]

        return Map.coord_to_xylists(vertices_trans)

    def get_map_bot_front(self, distance=None) -> list:
        if distance is None:
            distance = self.robot.length / 0.8
        vertices = [np.array([self.robot.position]), self.robot.get_bot_front(distance)]
        vertices_trans = [Map.coordtransform_world_to_map(v, self.arena_length, self.map_length) for v in
                          vertices]

        return Map.coord_to_xylists(vertices_trans)

    def update(self, display_front: bool = True) -> None:
        # clear display
        self.setColor(0x000000)
        self.fillRectangle(0, 0, self.height, self.width)

        self.setColor(0xFFFFFF)
        px, py = self.get_map_bot_vertices()
        self.drawPolygon(px, py)

        if display_front:
            px, py = self.get_map_bot_front()
            self.drawLine(px[0], py[0], px[1], py[1])
