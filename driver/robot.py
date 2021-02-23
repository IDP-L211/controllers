# Copyright (C) 2021 Weixuan Zhang, Eleanor Clifford, Jason Brown
#
# SPDX-License-Identifier: MIT
"""This module contains a class representing the robot.
"""

from controller import Robot
import numpy as np

from devices.sensors import IDPCompass, IDPGPS, IDPDistanceSensor
from devices.motors import IDPMotorController

from strategies.motion import MotionControlStrategies

from misc.utils import rotate_vector, get_target_bearing
from misc.mapping import Map


class IDPRobot(Robot):
    """Class representing the robot
    
    Attributes:
        compass (IDPCompass): The compass
        gps (IDPGPS): The GPS
        length (float): Length of the robot, parallel to the axis running back-to-front, in meters
        motors (IDPMotorController): The two motors
        target_bearing (float): Target bearing for the robot to turn to
        target_bearing_threshold (float): Threshold determining whether the target bearing is reached
        target_distance_threshold (float): Threshold determining whether the target coordinate is reached
        target_pos (list): Target coordinate to move to
        timestep (float): Time step of the current world
        ultrasonic (IDPDistanceSensor): The ultrasonic sensor
        width (float): Width of the robot, perpendicular to the axis running back-to-front
    """

    def __init__(self):
        super().__init__()

        self.length = 0.2
        self.width = 0.1

        self.timestep = int(self.getBasicTimeStep())

        # Devices
        self.gps = IDPGPS('gps', self.timestep)
        self.compass = IDPCompass('compass', self.timestep)
        self.ultrasonic = IDPDistanceSensor('ultrasonic', self.timestep)
        self.motors = IDPMotorController('wheel1', 'wheel2')

        # Where the bot is trying to path to
        self.target_pos = [None, None]
        self.target_distance_threshold = 0.1

        # If we need to point bot in a specific direction, otherwise it points at target if this is None
        # This would be interpreted as a bearing from north
        self.target_bearing = None
        self.target_bearing_threshold = np.pi / 50

    def getDevice(self, name: str):
        # here to make sure no device is retrieved this way
        if name in ['gps', 'compass', 'wheel1', 'wheel2', 'ultrasonic']:
            raise RuntimeError('Please use the corresponding properties instead')
        return Robot.getDevice(self, name)

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
            return True

        return abs(self.target_angle) <= self.target_bearing_threshold

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
            arena_length (float): Side length of the arena, which is assumed to be a square
            name (str): Name of the Display node, default to 'map'

        Returns:
            Map: The map
        """
        return Map(self, arena_length, name)

    def drive_to_position(self, target_pos: list) -> bool:
        """For this time step go to this position

        Args:
            target_pos ([float, float]): The East-North co-ords of the target position
        Returns:
            bool: If we are at our target
        """

        # Need to clear any target_bearing so it doesn't mess up target_angle
        self.target_bearing = None
        self.target_pos = target_pos
        self.motors.velocities = MotionControlStrategies.maximise_a_wheel_speed(self)

        return self.reached_target

    def face_bearing(self, target_bearing: float) -> bool:
        """For this time step go to this position

        Args:
            target_bearing (float): Desired bearing of our robot
        Returns:
            bool: If we are at our target
        """
        self.target_bearing = target_bearing
        self.motors.velocities = MotionControlStrategies.distance_angle_error(self)

        return self.reached_bearing

    def execute_action(self, actions: list) -> bool:
        """Execute the first action in a set of actions

        When each action is completed it's removed from the list. Using list mutability this allows us to alter / check
        the action list elsewhere in the code to see the bots progress and also change its objectives.
        If people have better ideas on how to do this I'm all ears.

        Actions:
            * move: Go to the given coordinates
            * turn: Face the given bearing

        Args:
            actions (list): Each list element is a tuple/list of ["action_type", value]

        Returns:
            bool: Whether action list is completed or not
        """
        # Check if action list is empty i.e. 'complete'
        if len(actions) == 0:
            return True

        # Execute action
        action_type = actions[0][0]
        action_value = actions[0][1]

        # Store the function associated with each action
        action_functions = {
            "move": self.drive_to_position,
            "face": self.face_bearing
        }

        # Check action is valid
        if action_type not in action_functions.keys():
            raise Exception(
                f"Action {action_type} is not a valid action, valid actions: {', '.join(action_functions.keys())}")

        # Execute action
        completed = action_functions[action_type](action_value)

        # If we completed this action we should remove it from our list
        if completed:
            del actions[0]

        return False
