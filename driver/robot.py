# Copyright (C) 2021 Weixuan Zhang, Tim Clifford, Jason Brown
#
# SPDX-License-Identifier: MIT
"""This module contains a class representing the robot.
"""

from controller import Robot
import numpy as np
import warnings


from devices.sensors import IDPCompass, IDPGPS, IDPDistanceSensor
from devices.motors import IDPMotorController

from strategies.motion import MotionControlStrategies

from misc.utils import rotate_vector, print_if_debug
from misc.mapping import Map


DEBUG = True


class IDPRobot(Robot):
    """Class representing the robot

    Attributes:
        compass (IDPCompass): The compass
        gps (IDPGPS): The GPS
        length (float): Length of the robot, parallel to the axis running back-to-front, in meters
        motors (IDPMotorController): The two motors
        target_bearing_threshold (float): Threshold determining whether the target bearing is reached
        target_distance_threshold (float): Threshold determining whether the target coordinate is reached
        timestep (float): Time step of the current world
        ultrasonic (IDPDistanceSensor): The ultrasonic sensor
        ir_long (IDPDistanceSensor): The IR sensor (long range)
        width (float): Width of the robot, perpendicular to the axis running back-to-front
    """

    def __init__(self):
        super().__init__()

        self.length = 0.2
        self.width = 0.1
        self.wheel_radius = 0.04

        self.timestep = int(self.getBasicTimeStep())

        # Devices
        self.gps = IDPGPS('gps', self.timestep)
        self.compass = IDPCompass('compass', self.timestep)
        self.ultrasonic = IDPDistanceSensor('ultrasonic', self.timestep)
        self.ir_long = IDPDistanceSensor('ir_long', self.timestep,
                                         decreasing=True, min_range=0.15)
        self.motors = IDPMotorController('wheel1', 'wheel2')

        # So we can cleanup if we change our action
        self.last_action_type = None
        self.last_action_value = None

        # Thresholds for finishing actions
        self.target_distance_threshold = 0.1
        self.target_bearing_threshold = np.pi / 100

        # For rotations
        self.rotation_angle = 0
        self.angle_rotated = 0
        self.last_bearing = None

        # For getting stuck
        self.stuck_last_step = False

    def getDevice(self, name: str):
        # here to make sure no device is retrieved this way
        if name in ['gps', 'compass', 'wheel1', 'wheel2', 'ultrasonic', 'ir_long']:
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

    def distance_from_bot(self, position) -> float:
        """The Euclidean distance between the bot and a position

        Args:
            position ([float, float]): Positions co-ordinates, East-North, m

        Returns:
            float: Distance between bot and target in metres
        """
        relative_position = np.array(position) - np.array(self.position)
        distance = np.hypot(*relative_position)
        return distance

    def angle_from_bot_from_bearing(self, bearing):
        """The clockwise angle from the direction our bot is facing to the bearing in radians

        Args:
            bearing (float): Bearing from north

        Returns:
            float: Angle measured clockwise from direction bot is facing, [-pi, pi]
        """
        angle = bearing - self.bearing

        # Need to adjust if outside [-pi,pi]
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi

        return angle

    def angle_from_bot_from_position(self, position) -> float:
        """The clockwise angle from the direction our bot is facing to the position in radians

        Args:
            position ([float, float]): Positions co-ordinates, East-North, m

        Returns:
            float: Angle measured clockwise from direction bot is facing, [-pi, pi]
        """
        relative_position = np.array(position) - np.array(self.position)
        bearing = np.arctan2(relative_position[0], relative_position[1])
        return self.angle_from_bot_from_bearing(bearing)

    def coordtransform_bot_to_world(self, vec: np.ndarray) -> np.ndarray:
        """Transform a position vector of a point in the robot frame (relative to the robot center)
        to the absolute position vector of that point in the world frame

        Args:
            vec(np.ndarray): A vector relative to the centre of the robot

        Returns:
            np.ndarray: The absolute position vector of the point in the world frame
        """
        # bearing is positive if clockwise while for rotation anticlockwistargete is positive
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

    def get_map(self, sensor, arena_length: float, name: str = 'map') -> Map:
        """Get a map of the arena, on which the current position and bounding box of the robot
        will be displayed.

        This requires the robot to have a Display child node with name 'map'.

        Args:
            sensor (IDPDistanceSensor): The distance sensor used on the robot
            arena_length (float): Side length of the arena, which is assumed to be a square
            name (str): Name of the Display node, default to 'map'

        Returns:
            Map: The map
        """
        return Map(self, sensor, arena_length, name)

    def reset_action_variables(self):
        """Cleanup method to be called when the current action changes. If executing bot commands manually
        (i.e. robot.drive_to_position), call this first.
        """
        self.rotation_angle = 0
        self.last_bearing = None
        self.angle_rotated = 0
        self.last_action_type = None
        self.last_action_value = None

    def drive_to_position(self, target_pos: list, reverse=False) -> bool:
        """Go to a position

        Args:
            target_pos ([float, float]): The East-North co-ords of the target position
            reverse (bool): Whether to reverse there
        Returns:
            bool: If we are at our target
        """

        distance = self.distance_from_bot(target_pos)
        angle = self.angle_from_bot_from_position(target_pos)

        reached_target = distance <= self.target_distance_threshold
        if reached_target:
            distance, angle = 0, 0

        # If we're reversing we change the angle so it mimics the bot facing the opposite way
        # When we apply the wheel velocities we negative them and voila we tricked the bot into reversing
        if reverse:
            angle = (np.sign(angle) * np.pi) - angle

        raw_velocities = MotionControlStrategies.angle_based_control(distance, angle)
        self.motors.velocities = raw_velocities if not reverse else -raw_velocities

        return reached_target

    def reverse_to_position(self, target_pos: list) -> bool:
        """Go to a position in reverse

        Args:
            target_pos ([float, float]): The East-North co-ords of the target position
        Returns:
            bool: If we are at our target
        """
        return self.drive_to_position(target_pos, reverse=True)

    def rotate(self, angle: float, rotation_rate=5.0) -> bool:
        """Rotate the bot a fixed angle at a fixed rate of rotation

        Args:
            angle (float): Angle to rotate in radians, positive is clockwise
            rotation_rate (float): Rate of rotation in radians per second, [0, 1]
        Returns:
            bool: If we completed rotation"""

        # First need to determine if this is a new rotation or a continued one
        if self.rotation_angle == 0:
            self.rotation_angle = angle
            self.last_bearing = self.bearing

        # Update how far we've rotated, making sure to correct if bearing crosses north
        self.angle_rotated -= self.angle_from_bot_from_bearing(self.last_bearing)
        self.last_bearing = self.bearing

        # Check if we're done
        angle_difference = self.rotation_angle - self.angle_rotated
        print(self.rotation_angle, self.angle_rotated, angle_difference)
        if abs(angle_difference) <= self.target_bearing_threshold:
            return True

        # Calculate angle_drive based on rotation rate
        turn_radius = self.width / 2
        angle_drive = (rotation_rate * turn_radius) / (self.motors.max_motor_speed * self.wheel_radius)

        if angle_drive > 1:
            max_rot = rotation_rate/angle_drive
            warnings.warn(f"Requested rotation rate of {rotation_rate} exceeds bot's apparent maximum of {max_rot}")

        self.motors.velocities = MotionControlStrategies.short_linear_region(0, angle_difference, angle_drive=angle_drive)
        return False

    def face_bearing(self, target_bearing: float) -> bool:
        """Face a given bearing

        Args:
            target_bearing (float): Desired bearing of our robot
        Returns:
            bool: If we are at our target
        """
        return self.rotate(self.angle_from_bot_from_bearing(target_bearing))

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
            self.motors.velocities = np.zeros(2)
            return True

        # Execute action
        action_type = actions[0][0]
        action_value = actions[0][1:]

        # Store the function associated with each action
        action_functions = {
            "move": self.drive_to_position,
            "face": self.face_bearing,
            "rotate": self.rotate,
            "reverse": self.reverse_to_position
        }

        # Check action is valid
        if action_type not in action_functions.keys():
            raise Exception(
                f"Action {action_type} is not a valid action, valid actions: {', '.join(action_functions.keys())}")

        # If we are changing our action we need to reset
        if action_type != self.last_action_type or action_value != self.last_action_value:
            self.reset_action_variables()

        # Update log of last action
        self.last_action_type = action_type
        self.last_action_value = action_value

        # Execute action
        completed = action_functions[action_type](*action_value)

        # If we completed this action we should remove it from our list
        if completed:
            self.reset_action_variables()
            print_if_debug(f"\nCompleted action: {actions[0]}", debug_flag=DEBUG)
            del actions[0]
            print_if_debug(f"Remaining actions:", debug_flag=DEBUG)

            # Check if action list is now empty
            if len(actions) == 0:
                print_if_debug("None", debug_flag=DEBUG)
                self.motors.velocities = np.zeros(2)
                return True

            print_if_debug('\n'.join(str(x) for x in actions), debug_flag=DEBUG)

        # Check if bot is stuck, note we only reach here if action not completed
        if abs(self.speed) <= 0.001:
            if self.stuck_last_step:
                print_if_debug("BOT STUCK - REVERSING", debug_flag=DEBUG)
                un_stuck_action = "reverse" if action_type != "reverse" else "move"
                actions.insert(0, (un_stuck_action, list(self.coordtransform_bot_to_world(np.array([0, -0.2])))))
                self.stuck_last_step = False
            else:
                self.stuck_last_step = True

        return False
