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

from misc.utils import rotate_vector, get_min_distance_rectangles, print_if_debug
from misc.mapping import Map
from misc.detection_handler import ObjectDetectionHandler
from misc.pid import PID

DEBUG = False


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
        ultrasonic_left (IDPDistanceSensor): The ultrasonic sensor on the left
        ultrasonic_right (IDPDistanceSensor): The ultrasonic sensor on the right
        infrared (IDPDistanceSensor): The IR sensor (long range)
        width (float): Width of the robot, perpendicular to the axis running back-to-front
    """

    def __init__(self):
        super().__init__()

        # note the sensors are assumed to be at the centre of the robot, and the robot is assumed symmetrical
        self.length = 0.2
        self.width = 0.1
        self.wheel_radius = 0.04
        self.colour = "red"

        self.arena_length = 2.4
        self.timestep = int(self.getBasicTimeStep())

        # Devices
        self.gps = IDPGPS('gps', self.timestep)
        self.compass = IDPCompass('compass', self.timestep)
        self.ultrasonic_left = IDPDistanceSensor('ultrasonic_left', self.timestep)
        self.ultrasonic_right = IDPDistanceSensor('ultrasonic_right', self.timestep)
        self.infrared = IDPDistanceSensor('infrared', self.timestep,
                                         decreasing=True, min_range=0.15)
        self.motors = IDPMotorController('wheel1', 'wheel2')

        # To store and process detections
        self.object_detection_handler = ObjectDetectionHandler()

        # Store internal action queue
        self.action_queue = []

        # Store the function associated with each action
        self.action_functions = {
            "move": self.drive_to_position,
            "face": self.face_bearing,
            "rotate": self.rotate,
            "reverse": self.reverse_to_position,
            "collect": self.collect_block
        }

        # So we can cleanup if we change our action
        self.last_action_type = None
        self.last_action_value = None

        # Thresholds for finishing actions
        self.target_distance_threshold = 0.01
        self.target_bearing_threshold = np.pi / 1000

        # For rotations
        self.rotation_angle = 0
        self.angle_rotated = 0
        self.last_bearing = None

        # For getting stuck
        self.stuck_last_step = False

        # Motion control
        self.timestep_actual = self.timestep / 1000  # Webots timestep is in ms
        self.pid_f_velocity = PID(1, 0, 0, self.timestep_actual)
        self.pid_r_velocity = PID(1, 0, 0, self.timestep_actual)
        self.pid_distance = PID(1, 0, 0, self.timestep_actual)
        self.pid_angle = PID(1.8, 0, 0.1, self.timestep_actual)

    def getDevice(self, name: str):
        # here to make sure no device is retrieved this way
        if name in ['gps', 'compass', 'wheel1', 'wheel2', 'ultrasonic_left', \
                    'ultrasonic_right', 'infrared']:
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
        relative_position = np.asarray(position) - np.asarray(self.position)
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
        relative_position = np.asarray(position) - np.asarray(self.position)
        bearing = np.arctan2(*relative_position)
        return self.angle_from_bot_from_bearing(bearing)

    def coordtransform_bot_cartesian_to_world(self, vec: np.ndarray) -> np.ndarray:
        """Transform a position vector of a point in the robot frame (relative to the robot center)
        to the absolute position vector of that point in the world frame

        Args:
            vec(np.ndarray): A vector relative to the centre of the robot

        Returns:
            np.ndarray: The absolute position vector of the point in the world frame
        """
        # bearing is positive if clockwise while for rotation anticlockwise is positive
        return np.asarray(self.position) + rotate_vector(vec, -self.bearing)

    def coordtransform_bot_polar_to_world(self, distance: float, angle: float) -> np.ndarray:
        """Given a distance and angle from our bot, return the objects position

        Args:
            distance (float): Distance to object in m
            angle (float): Angle from bot to object in rads

        Returns:
            [float, float]: Positions co-ordinates, East-North, m
        """
        bot_cartesian = np.array([distance * np.sin(angle), distance * np.cos(angle)])
        return self.coordtransform_bot_cartesian_to_world(bot_cartesian)

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

        return list(map(self.coordtransform_bot_cartesian_to_world, center_to_corners))

    def get_bot_front(self, distance: float) -> np.ndarray:
        """Get the coordinates of a point a certain distance in front of the center of the robot

        Args:
            distance (float): Distance in front of the center

        Returns:
            np.ndarray: The coordinate
        """

        return self.coordtransform_bot_cartesian_to_world(np.array([0, distance]))

    def get_sensor_distance_to_wall(self):
        """Get the distance the sensor should measure if there is nothing between it and the wall

        Note that the sensor is assumed to be at the centre of the robot.

        Returns:
            float: The distance
        """
        wall_from_origin = self.arena_length / 2

        sin_bearing = np.sin(self.bearing)
        distance_from_x = abs(
            (wall_from_origin - np.sign(sin_bearing) * self.position[0]) / sin_bearing
        ) if sin_bearing != 0 else np.inf

        cos_bearing = np.cos(self.bearing)
        distance_from_y = abs(
            (wall_from_origin - np.sign(cos_bearing) * self.position[1]) / cos_bearing
        ) if cos_bearing != 0 else np.inf

        return min(distance_from_x, distance_from_y)

    def get_min_distance_vertex_to_wall(self):
        """Get the minimum distance from any of the vertices of the robot to the wall

        Returns:
            float: The minimum distance
        """
        vertices = self.get_bot_vertices()
        max_x_abs = max(map(lambda v: abs(v[0]), vertices))
        max_y_abs = max(map(lambda v: abs(v[1]), vertices))

        wall_from_origin = self.arena_length / 2
        return min(wall_from_origin - max_x_abs, wall_from_origin - max_y_abs)

    def get_min_distance_bot_to_bot(self, other_bot_vertices: list) -> float:
        """Get the minimum distances between two robots

        Args:
            other_bot_vertices ([np.ndarray]): List of vertices of the other robot, needs to be specified in
            a clockwise order

        Returns:
            float: The minimum distance between the two robots
        """
        return get_min_distance_rectangles(self.get_bot_vertices(), other_bot_vertices)

    def get_map(self, sensor: IDPDistanceSensor, name: str = 'map') -> Map:
        """Get a map of the arena, on which the current position and bounding box of the robot
        will be displayed.

        This requires the robot to have a Display child node with name 'map'.

        Args:
            sensor (IDPDistanceSensor): The distance sensor used on the robot
            name (str): Name of the Display node, default to 'map'

        Returns:
            Map: The map
        """
        return Map(self, sensor, self.arena_length, name)

    def reset_action_variables(self):
        """Cleanup method to be called when the current action changes. If executing bot commands manually
        (i.e. robot.drive_to_position), call this first.
        """
        self.rotation_angle = 0
        self.last_bearing = None
        self.angle_rotated = 0
        self.last_action_type = None
        self.last_action_value = None
        self.pid_f_velocity.reset()
        self.pid_r_velocity.reset()
        self.pid_distance.reset()
        self.pid_angle.reset()

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
            return True

        # If we're reversing we change the angle so it mimics the bot facing the opposite way
        # When we apply the wheel velocities we negative them and voila we tricked the bot into reversing
        if reverse:
            angle = (np.sign(angle) * np.pi) - angle

        # raw_velocities = MotionControlStrategies.angle_based_control(distance, angle)
        raw_velocities = MotionControlStrategies.combined_pid(current_f_velocity=self.speed, current_distance=distance,
                                                              current_angle=angle, pid_f_velocity=self.pid_f_velocity,
                                                              pid_distance=self.pid_distance, pid_angle=self.pid_angle)
        self.motors.velocities = raw_velocities if not reverse else -raw_velocities
        return False

    def reverse_to_position(self, target_pos: list) -> bool:
        """Go to a position in reverse

        Args:
            target_pos ([float, float]): The East-North co-ords of the target position
        Returns:
            bool: If we are at our target
        """
        return self.drive_to_position(target_pos, reverse=True)

    def rotate(self, angle: float, rotation_rate=None) -> bool:
        """Rotate the bot a fixed angle at a fixed rate of rotation

        Args:
            angle (float): Angle to rotate in radians, positive is clockwise
            rotation_rate (float): Rate of rotation in radians per second, [0, 1], None will cause PID to use max (8.0)
        Returns:
            bool: If we completed rotation
        """

        # First need to determine if this is a new rotation or a continued one
        if self.rotation_angle == 0:
            self.rotation_angle = angle
            self.last_bearing = self.bearing

        # Update how far we've rotated
        angle_rotated_in_timestep = -self.angle_from_bot_from_bearing(self.last_bearing)
        self.angle_rotated += angle_rotated_in_timestep
        r_velocity = angle_rotated_in_timestep / self.timestep_actual
        self.last_bearing = self.bearing

        # Check if we're done
        angle_difference = self.rotation_angle - self.angle_rotated
        if abs(angle_difference) <= self.target_bearing_threshold and abs(angle_rotated_in_timestep) * 5 <= self.target_bearing_threshold:
            return True

        velocities = MotionControlStrategies.combined_pid(current_r_velocity=r_velocity, current_angle=angle_difference,
                                                          required_r_velocity=rotation_rate, pid_angle=self.pid_angle,
                                                          pid_r_velocity=self.pid_r_velocity, switch_angle=np.pi*10)
        self.motors.velocities = velocities
        return False

    def face_bearing(self, target_bearing: float) -> bool:
        """Face a given bearing

        Args:
            target_bearing (float): Desired bearing of our robot
        Returns:
            bool: If we are at our target
        """
        return self.rotate(self.angle_from_bot_from_bearing(target_bearing))

    def collect_block(self, block_pos):
        """Collect block at position

        Args:
            block_pos ([float, float]): The East-North co-ords of the blocks position
        Returns:
            bool: If we are at our target
        """

        # Update these variables when we have more info
        distance_from_block_to_stop = 0.1
        rotate_angle = np.pi
        home_pos = [0, 0]

        # Calculate pos to got to to be near block not on it
        distance = self.distance_from_bot(block_pos) - distance_from_block_to_stop
        target_pos = self.coordtransform_bot_polar_to_world(distance, self.angle_from_bot_from_position(block_pos))

        # Need to add action that deposits block
        actions = [
            ("move", target_pos),
            ("rotate", rotate_angle),
            ("move", home_pos)
        ]

        self.action_queue = [self.action_queue[0]] + actions + self.action_queue[1:]
        return True

    def do(self, *args):
        """Small wrapper to make telling robot to do something a little cleaner"""
        self.action_queue = [args]

    def execute_next_action(self) -> bool:
        """Execute the next action in self.action_queue

        When each action is completed it's removed from the list. Using list mutability this allows us to alter / check
        the action list elsewhere in the code to see the bots progress and also change its objectives.
        If people have better ideas on how to do this I'm all ears.

        Returns:
            bool: Whether action list is completed or not
        """

        # Check if action list is empty i.e. 'complete'
        if len(self.action_queue) == 0:
            self.motors.velocities = np.zeros(2)
            return True

        # Execute action
        action_type = self.action_queue[0][0]
        action_value = self.action_queue[0][1:]

        # Check action is valid
        if action_type not in self.action_functions.keys():
            raise Exception(
                f"Action {action_type} is not a valid action, valid actions: {', '.join(self.action_functions.keys())}")

        # If we are changing our action we need to reset
        if action_type != self.last_action_type or action_value != self.last_action_value:
            self.reset_action_variables()

        # Update log of last action
        self.last_action_type = action_type
        self.last_action_value = action_value

        # Execute action
        completed = self.action_functions[action_type](*action_value)

        # If we completed this action we should remove it from our list
        if completed:
            self.reset_action_variables()
            print_if_debug(f"\nCompleted action: {self.action_queue[0]}", debug_flag=DEBUG)
            del self.action_queue[0]
            print_if_debug(f"Remaining actions:", debug_flag=DEBUG)

            # Check if action list is now empty
            if len(self.action_queue) == 0:
                print_if_debug("None", debug_flag=DEBUG)
                self.motors.velocities = np.zeros(2)
                return True

            print_if_debug('\n'.join(str(x) for x in self.action_queue), debug_flag=DEBUG)

        # Check if bot is stuck, note we only reach here if action not completed
        if abs(self.speed) <= 0.001:
            if self.stuck_last_step:
                print_if_debug("BOT STUCK - REVERSING", debug_flag=DEBUG)
                un_stuck_action = "reverse" if action_type != "reverse" else "move"
                self.action_queue.insert(0, (un_stuck_action,
                                             list(self.coordtransform_bot_cartesian_to_world(np.array([0, -0.1])))))
                self.stuck_last_step = False
            else:
                self.stuck_last_step = True

        return False

    def log_object_detection(self, position, classification="unknown"):
        """Take an object detected a certain distance away and store its detection

        Args:
            position ([float, float]):  Positions co-ordinates in world, East-North, m
            classification (string): What we think the object is
        """
        self.object_detection_handler.new_detection(position, classification)

    def get_target(self) -> list:
        """Decide on a new target block for robot

        If targeting is not just get closest block there could be more logic here, potentially calls to a script in
            strategies folder that applied more complex algorithms and could even return a list of ordered targets

        For now we just choose the closest block of correct colour or unknown colour

        Returns:
            [float, float]: Targets co-ordinates, East-North, m
        """
        valid_classes = ["box", f"{self.colour}_box"]
        object_list = self.object_detection_handler.get_sorted_objects(
            valid_classes=valid_classes,
            key=lambda target: self.distance_from_bot(target.postion)
        )

        return object_list[0].position
