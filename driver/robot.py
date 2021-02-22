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
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        # Sensors
        self.gps = self.getDevice('gps')  # or use createGPS() directly
        self.compass = self.getDevice('compass')

        # Where the bot is trying to path to
        self.target_pos = [None, None]
        self.target_distance_threshold = 0.1

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
        distance = np.hypot(*distance_vector)
        return distance

    @property
    def wheel_speeds1(self) -> list:
        """Set wheel speeds based on angle and distance error

        This method performs well for face_bearing but has issues with point to point travel - slow turn times and
        slowing down a lot when near target

        Returns:
            [float, float]: The speed for the left and right motors respectively to correct both angle and distance
                            error. Given as a fraction of max speed.
        """
        # Control parameters
        k_p_forward = 4
        k_p_rotation = 4

        forward_speed = self.target_distance * k_p_forward
        rotation_speed = self.target_angle * k_p_rotation

        # If we are within distance threshold we no longer need to move forward
        forward_speed = 0 if self.reached_target else forward_speed

        # If we are within angle threshold we should stop turning
        if self.target_bearing is None:
            rotation_speed = 0 if self.reached_target else rotation_speed
        else:
            rotation_speed = 0 if self.reached_bearing else rotation_speed

        # Attenuate and possibly reverse forward speed based on angle of bot - don't want to speed off away from it
        forward_speed *= np.cos(self.target_angle)

        # Combine speeds
        speeds = np.array([forward_speed + rotation_speed, forward_speed - rotation_speed])

        # This might be above our motor maximums so we'll use sigmoid to normalise our speeds to this range
        # Sigmoid bounds -inf -> inf to 0 -> 1 so we'll need to do some correcting
        speeds = (1/(1 + np.exp(-speeds))) * 2 - 1

        # Alternative to using sigmoid
        # speeds = np.tanh(speeds)

        # Another alternative where the minimum motor speed is half its value
        # speeds = (1 / (1 + np.exp(-speeds))) + ((np.sign(speeds) - 1) * 0.5)

        return list(speeds)

    @property
    def wheel_speeds2(self):
        """Set fastest wheel speed to maximum with the other wheel slowed to facilitate turning

        By using a cos^2 for our forward speed and sin^2 for our rotation speed, the fastest wheel will always be at max
        drive fraction and the other wheel will correct for angle

        This method seems to perform better for point to point travel but is slow to orientate the bot in face_bearing

        Returns:
            [float, float]: The speed for the left and right motors respectively to correct both angle and distance
                            error. Given as a fraction of max speed.
        """
        # For some reason (probably floating point errors), we occasionally get warnings about requested speed exceeding
        # max velocity even though they are equal. We shall subtract a small quantity to avoid this annoyance.
        small_speed = 1e-5

        # How fast our fastest wheel should go
        forward_speed = (np.cos(self.target_angle)**2) - small_speed if abs(self.target_angle) <= np.pi / 2 else 0

        # How much slower our slower wheel should go for turning purposes
        rotation_speed = (np.sin(self.target_angle)**2) if abs(self.target_angle) <= np.pi / 2 else 1

        # If we are within distance threshold we no longer need to move forward
        forward_speed = 0 if self.reached_target else forward_speed

        # If we are within angle threshold we should stop turning
        if self.target_bearing is None:
            rotation_speed = 0 if self.reached_target else rotation_speed
        else:
            rotation_speed = 0 if self.reached_bearing else rotation_speed

        # Combine our speeds based on the sign of our angle
        if self.target_angle >= 0:
            return [forward_speed + rotation_speed, forward_speed - rotation_speed]
        else:
            return [forward_speed - rotation_speed, forward_speed + rotation_speed]

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

    def set_motor_velocities(self, left, right):
        """Set the velocities for each motor

        Args:
            left: float: Speed for left wheel, fraction of max speed (-1 -> 1)
            right: float: Speed for right wheel, fraction of max speed (-1 -> 1)
        """

        left_speed = left * self.max_motor_speed
        right_speed = right * self.max_motor_speed

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

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

        # Need to clear any target_bearing so it doesn't mess up target_angle
        self.target_bearing = None
        self.target_pos = target_pos
        self.set_motor_velocities(*self.wheel_speeds2)

        return self.reached_target

    def face_bearing(self, target_bearing: float) -> bool:
        """For this time step go to this position

        Args:
            target_bearing: float: Desired bearing of our robot
        Returns:
            bool: If we are at our target
        """
        self.target_bearing = target_bearing
        self.set_motor_velocities(*self.wheel_speeds1)

        return self.reached_bearing

    def execute_actions(self, actions: list) -> bool:
        """Execute a set of actions from a list in order

        When each action is completed it's removed from the list. Using list mutability this allows us to alter / check
        the action list elsewhere in the code to see the bots progress and also change its objectives.
        If people have better ideas on how to do this I'm all ears.

        Args:
            actions: list: If an element is a list it's treated as co-ords, if it's a float it's treated as a bearing

        Returns:
            bool: Whether action list is completed or not
        """

        # Check if action list is empty i.e. 'complete'
        if len(actions) == 0:
            return True

        # Execute action
        else:
            action = actions[0]

            # Action is a list and therefore co-ords to drive to
            if isinstance(action, list):
                if len(action) != 2:
                    raise Exception(f"Action {action} given to robot does not have 2 elements")
                else:
                    completed = self.drive_to_position(action)

            # Action is a float and therefore a bearing to face
            elif isinstance(action, float):
                completed = self.face_bearing(action)

            else:
                raise Exception(f"Unrecognised action type for {action}, should be list or float")

            # If we completed this action we should remove it from our list
            if completed:
                del actions[0]

            return False
