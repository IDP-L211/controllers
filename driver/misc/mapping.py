# Copyright (C) 2021 Weixuan Zhang
#
# SPDX-License-Identifier: MIT
"""This module contains classes for displaying a real time map of the arena
from sensor data.
"""

import numpy as np

from controller import Display


class Map(Display):
    def __init__(self, robot, sensor, arena_length: float, name: str = 'map'):
        super().__init__(name)
        self.robot = robot
        self.sensor = sensor
        self.arena_length = arena_length
        self.width = self.getWidth()
        self.height = self.getHeight()
        self.map_length = min(self.width, self.height)

        self.plot_commands = []

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

    def coordtransform_world_to_map(self, vec: list) -> list:
        """Transfer the world coordinate (East, North) to image coordinate

        The image coordinate has (0,0) at the top left corner and (width-1,height-1) at
        the bottom right corner. Therefore, the North coordinate needs to be reversed.
        The origin also need to be shifted to the center.

        Args:
            vec(list): The coordinate in world frame

        Returns:
            [int, int]: The coordinate on the map, integer number of pixels
        """
        vec = np.array(vec)
        vec_trans = (Map.convert_unit(vec, self.arena_length, self.map_length) * np.array([1, -1])
                     + np.array([self.map_length] * 2) / 2)
        # convert to integer, must use toList() to ensure type as int instead of np.int64
        return vec_trans.astype(int).tolist()

    def get_map_bot_vertices(self) -> list:
        # formatted as [[x1, x2, x3...], [y1, y2, y3...]]
        return Map.coord_to_xylists(list(map(
            self.coordtransform_world_to_map,
            self.robot.get_bot_vertices()  # vertices in world frame
        )))

    def get_map_bot_front(self, distance: float = 0) -> list:
        if distance <= 0:
            distance = self.robot.length / 0.8

        return self.coordtransform_world_to_map(self.robot.get_bot_front(distance))

    def draw_marker(self, map_coord: list) -> None:
        # must be of type int not np.int64, pass in a list instead of np.ndarray
        self.fillOval(*map_coord, 3, 3)

    def draw_line_from_botcenter(self, map_coord: list) -> None:
        # must be of type int not np.int64, pass in a list instead of np.ndarray
        self.drawLine(*self.coordtransform_world_to_map(self.robot.position), *map_coord)

    def plot_coordinate(self, world_coord):
        self.plot_commands.append(lambda: self.draw_marker(self.coordtransform_world_to_map(world_coord)))

    def clear(self):
        # clear display
        self.setColor(0x000000)
        self.fillRectangle(0, 0, self.height, self.width)

    def update(self, clear_frame: bool = True, draw_robot: bool = True, draw_line: bool = True,
               draw_range: bool = True) -> None:
        if clear_frame:
            self.clear()

        if draw_robot:
            # draw bounding box
            self.setColor(0xFFFFFF)
            self.drawPolygon(*self.get_map_bot_vertices())

        # draw markers
        front_coord = self.get_map_bot_front(self.sensor.getValue())
        if draw_line:
            self.draw_line_from_botcenter(front_coord)
        self.draw_marker(front_coord)
        if draw_range:
            self.draw_marker(self.get_map_bot_front(self.sensor.max_range))

        # other external plot commands
        while len(self.plot_commands) > 0:
            self.plot_commands.pop(0)()
