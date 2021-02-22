# Copyright (C) 2021 Weixuan Zhang
#
# SPDX-License-Identifier: MIT
"""This module contains classes representing the robot.
"""

from math import atan2, pi

from controller import Robot, GPS, Compass, Motor
import numpy as np

from utils import rotate_vector, get_target_bearing
from mapping import Map


class IDPCompass(Compass):
    def __init__(self, name, sampling_rate):
        super().__init__(name)
        self.enable(sampling_rate)


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
        self.max_motor_speed = min(self.left_motor.getMaxVelocity(), self.right_motor.getMaxVelocity())

        # Sensors
        self.gps = self.getDevice('gps')  # or use createGPS() directly
        self.compass = self.getDevice('compass')

        # Where the bot is trying to path to
        self.target_pos = [None, None]
        self.target_distance_threshold = 0.2

        # If we need to point bot in a specific direction, otherwise it points at target if this is None
        # This would be interpreted as a bearing from north
        self.target_bearing = None
        self.target_bearing_threshold = np.pi / 50

    # .getDevice() will call createXXX if the tag name is not in __devices[]
    def createCompass(self, name: str) -> IDPCompass:  # override method to use the custom Compass class
        return IDPCompass(name, self.timestep)

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
        rad = atan2(north[0], north[2])  # [-pi, pi]
        return rad + 2 * pi if rad < 0 else rad

    @property
    def target_angle(self) -> float:
        """The clockwise angle from the direction our bot is facing to the target in radians

        Returns:
            float: Angle measured clockwise from direction bot is facing, [-pi, pi]
        """
        target_bearing = get_target_bearing(self.position, self.target_pos) if self.target_bearing is None \
            else self.target_bearing
        angle = target_bearing - self.bearing

        # Need to adjust if outside [-pi,pi]
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi

        return angle

    @property
    def target_distance(self) -> float:
        """The Euclidean distance between the bot and its target

        Returns:
            float: Distance between bot and target in metres
        """
        if None in self.target_pos:
            return 0

        distance_vector = np.array(self.target_pos) - np.array(self.position)
        distance = np.sqrt(sum(x**2 for x in distance_vector))
        return distance

    @property
    def wheel_rotation_speed(self) -> list:
        """Uses control theory to calculate the speeds for the wheels to spin in opposite directions to correct angle
        error

        Currently implemented as proportional control

        Returns:
            [float, float]: The speed for the left and right motors respectively to correct angle error
        """
        # Control
        k_p = 1
        speed = self.target_angle * k_p

        # If we are within the threshold we should stop turning to face target, unless we have override
        if self.target_bearing is None:
            speed = 0 if self.reached_target else speed
        else:
            speed = 0 if self.reached_bearing else speed

        return [speed, -speed]

    @property
    def wheel_forward_speed(self) -> list:
        """Uses control theory to calculate the speeds for the wheels to spin in the same direction to correct distance
        error

        Currently implemented as proportional control

        Returns:
            [float, float]: The speed for the left and right motors respectively to correct distance error
        """

        # Control
        k_p = 1
        speed = self.target_distance * k_p

        # If we are within the threshold we no longer need to move forward
        speed = 0 if self.reached_target else speed

        # We need to attenuate based on angle so we don't drive away from target
        # Will also reverse robot if it's facing backwards
        speed *= np.cos(self.target_angle)
        return [speed, speed]

    @property
    def wheel_speeds(self) -> list:
        """Uses wheel rotation and forward speeds to calculate the overall speeds for the wheels to reduce angle and
        distance error

        Returns:
            [float, float]: The speed for the left and right motors respectively to correct both angle and distance
                            error
        """
        speeds = np.array(self.wheel_forward_speed) + np.array(self.wheel_rotation_speed)

        # This might be above our motor maximums so we'll use sigmoid to normalise our speeds to this range
        # Sigmoid bounds -inf -> inf to 0 -> 1 so we'll need to do some correcting
        speeds = (1/(1 + np.exp(-speeds)))
        speeds = (speeds * 2) - 1
        speeds *= self.max_motor_speed

        return list(speeds)

    @property
    def reached_target(self) -> bool:
        """Whether we are at our target

         Returns:
             bool: If we are within the threshold for our target
        """
        return self.target_distance <= self.target_distance_threshold

    @property
    def reached_bearing(self) -> bool:
        """If we provide a bearing override, whether we are within tolerance

         Returns:
             bool: If we are within the threshold for our bearing
        """
        if self.target_bearing is None:
            reached_angle = None
        else:
            reached_angle = abs(self.target_angle) <= self.target_bearing_threshold
        return reached_angle

    def set_motor_velocities(self):
        """Set the velocities for each motor according to wheel_speeds"""
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(self.wheel_speeds[0])
        self.right_motor.setVelocity(self.wheel_speeds[1])

    def get_bot_vertices(self):
        """Get the coordinates of vertices of the bot in world frame (i.e. in meters)

        The robot is assumed to be a rectangle.

        Returns:
            [np.ndarray]: List of coordinates
        """
        center = np.array(self.position)
        angle = -self.bearing  # bearing is positive if clockwise while for rotation anticlockwise is positive
        center_to_corner = np.array([self.width, self.length]) / 2

        center_to_topleft = center_to_corner * np.array([-1, 1])
        center_to_topright = center_to_corner
        center_to_bottomright = center_to_corner * np.array([1, -1])
        center_to_bottomleft = -center_to_corner

        center_to_corners = [center_to_topleft, center_to_topright,
                             center_to_bottomright, center_to_bottomleft]

        vertices = [center + rotate_vector(v, angle) for v in center_to_corners]
        return vertices

    def get_bot_front(self, distance: float):
        """Get the coordinates of a point a certain distance in front of the center of the robot

        Args:
            distance (float): Distance in front of the center

        Returns:
            np.ndarray: The coordinate
        """

        return np.array(self.position) + rotate_vector(np.array([0, distance]), -self.bearing)

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

    def drive_to_position(self, target_pos: list) -> bool:
        """For this time step go to this position

        Args:
            target_pos: [float, float]: The East-North co-ords of the target position
        Returns:
            bool: If we are at our target
        """
        self.target_pos = target_pos
        self.set_motor_velocities()
        return self.reached_target

    def face_bearing(self, target_bearing: float) -> bool:
        """For this time step go to this position

        Args:
            target_bearing: float: Desired bearing of our robot
        Returns:
            bool: If we are at our target
        """
        self.target_bearing = target_bearing
        self.set_motor_velocities()
        return self.reached_bearing
