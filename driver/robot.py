# Copyright (C) 2021 Weixuan Zhang
#
# SPDX-License-Identifier: MIT
"""This module contains classes representing the robot.
"""

from controller import Robot, GPS, Compass, Motor, DistanceSensor
import numpy as np

from utils import rotate_vector
from mapping import Map


class IDPCompass(Compass):
    def __init__(self, name, sampling_rate):
        super().__init__(name)
        self.enable(sampling_rate)


class IDPDistanceSensor(DistanceSensor):
    def __init__(self, name, sampling_rate):
        super().__init__(name)
        self.enable(sampling_rate)
        self.max_range = self.getLookupTable()[-3]


class IDPGPS(GPS):
    def __init__(self, name, sampling_rate):
        super().__init__(name)  # default to infinite resolution
        if self.getCoordinateSystem() != 0:
            raise RuntimeWarning('Need to set GPS coordinate system in WorldInfo to local')
        self.enable(sampling_rate)


class IDPMotor(Motor):
    def __init__(self, name):
        super().__init__(name)
        self.setPosition(float('inf'))
        self.setVelocity(0.0)


class IDPRobot(Robot):
    def __init__(self):
        super().__init__()

        """
        length: Length of the robot, parallel to the axis running back-to-front, in meters
        width: Width of the robot, perpendicular to the axis running back-to-front
        """
        self.length = 0.2
        self.width = 0.1

        self.timestep = int(self.getBasicTimeStep())  # get the time step of the current world.

        # Motors
        self.left_motor = self.getDevice('wheel1')
        self.right_motor = self.getDevice('wheel2')

        # Sensors
        self.gps = self.getDevice('gps')  # or use createGPS() directly
        self.compass = self.getDevice('compass')
        self.ultrasonic = self.getDevice('ultrasonic')

    # .getDevice() will call createFoo if the tag name is not in __devices[]
    def createCompass(self, name: str) -> IDPCompass:  # override method to use the custom Compass class
        return IDPCompass(name, self.timestep)

    def createDistanceSensor(self, name: str) -> IDPDistanceSensor:
        return IDPDistanceSensor(name, self.timestep)

    def createGPS(self, name: str) -> IDPGPS:
        return IDPGPS(name, self.timestep)

    def createMotor(self, name: str) -> IDPMotor:
        return IDPMotor(name)

    # The coordinate system is NUE, meaning positive-x is North etc
    @property
    def position(self) -> list:
        """The current position of the robot

        Returns:
            [float, float]: Position (East, North) in meters
        """
        pos = self.gps.getValues()
        return [pos[2], pos[0]]  # (East, North) so extract z and x

    @property
    def speed(self) -> float:
        """The current speed of the robot measured by the GPS

        Returns:
            float: Current speed (ms^-1)
        """
        return self.gps.getSpeed()

    @property
    def bearing(self) -> float:
        """The current bearing of the robot in radians

        Returns:
            float: Bearing measured clockwise from North, [0, 2pi]
        """
        north = self.compass.getValues()
        if np.isnan([north[0], north[2]]).any():
            raise RuntimeError('No data from Compass, make sure "xAxis" and "yAxis" are set to TRUE')
        rad = np.arctan2(north[0], north[2])  # [-pi, pi]
        return rad + 2 * np.pi if rad < 0 else rad

    def coordtransform_bot_to_world(self, vec: np.ndarray) -> np.ndarray:
        """Transform a position vector of a point in the robot frame (relative to the robot center)
        to the absolute position vector of that point in the world frame

        Args:
            vec(np.ndarray): A vector relative to the centre of the robot

        Returns:
            np.ndarray: The absolute position vector of the point in the world frame
        """
        # bearing is positive if clockwise while for rotation anticlockwise is positive
        return np.array(self.position) + rotate_vector(vec, -self.bearing)

    def get_bot_vertices(self) -> list:
        """Get the coordinates of vertices of the bot in world frame (i.e. in meters)

        The robot is assumed to be a rectangle.

        Returns:
            [np.ndarray]: List of coordinates
        """

        center_to_corner = np.array([self.width, self.length]) / 2

        center_to_topleft = center_to_corner * np.array([-1, 1])
        center_to_topright = center_to_corner
        center_to_bottomright = center_to_corner * np.array([1, -1])
        center_to_bottomleft = -center_to_corner

        center_to_corners = [center_to_topleft, center_to_topright,
                             center_to_bottomright, center_to_bottomleft]

        return list(map(self.coordtransform_bot_to_world, center_to_corners))

    def get_bot_front(self, distance: float) -> np.ndarray:
        """Get the coordinates of a point a certain distance in front of the center of the robot

        Args:
            distance (float): Distance in front of the center

        Returns:
            np.ndarray: The coordinate
        """

        return self.coordtransform_bot_to_world(np.array([0, distance]))

    def get_map(self, arena_length: float, name: str = 'map') -> Map:
        """Get a map of the arena, on which the current position and bounding box of the robot
        will be displayed.

        This requires the robot to have a Display child node with name 'map'.

        Args:
            arena_length(float): Side length of the arena, which is assumed to be a square
            name(str): Name of the Display node, default to 'map'

        Returns:
            Map: The map
        """
        return Map(self, arena_length, name)

    def drive_to_position(self, target):
        pass
