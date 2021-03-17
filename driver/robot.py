# Copyright (C) 2021 Weixuan Zhang, Eleanor Clifford, Jason Brown
#
# SPDX-License-Identifier: MIT
"""This module contains a class representing the robot.
"""

from typing import Union
from itertools import chain
from operator import attrgetter
from enum import Enum

from controller import Robot
import numpy as np

from devices.sensors import IDPCompass, IDPGPS, IDPDistanceSensor, IDPColorDetector
from devices.motors import IDPMotorController, IDPGate
from devices.radio import IDPRadio

from modules.motion import MotionCS
from modules.utils import print_if_debug, ensure_list_or_tuple, fire_and_forget
from modules.geometry import rotate_vector, get_rectangle_sides, get_min_distance_point_rectangle, \
    get_min_distance_rectangles, point_in_rectangle
from modules.mapping import Map
from modules.pid import PID, DataRecorder
from modules.targeting import TargetingHandler, Target, TargetCache

DEBUG_ACTIONS = False
DEBUG_COLLISIONS = True
DEBUG_COLLECT = True
DEBUG_TARGETS = False
DEBUG_SCAN = True
DEBUG_STUCK = False
DEBUG_OBSTRUCTIONS = False
tau = np.pi * 2


class IDPRobotState(Enum):
    START_COLLECT                          = 0
    APPROACHING_TARGET_FROM_CENTER         = 1
    ROTATING_TO_FACE_TARGET_AFTER_APPROACH = 2
    DRIVING_TO_TARGET                      = 3
    ROTATING_TO_FACE_TARGET_FOR_DETECT     = 4
    DETECTING_COLOUR                       = 5
    BRAKING                                = 6
    GATE_OPENING                           = 7
    CHECKING_ROTATE                        = 8
    ROTATING_TO_COLLECT                    = 9
    GATE_CLOSING                           = 10


class IDPRobot(Robot):
    """Class representing the robot

    Attributes:
        compass (IDPCompass): The compass
        gps (IDPGPS): The GPS
        motors (IDPMotorController): The two motors
        default_target_bearing_threshold (float): Threshold determining whether the target bearing is reached
        default_target_distance_threshold (float): Threshold determining whether the target coordinate is reached
        timestep (float): Time step of the current world
        ultrasonic_left (IDPDistanceSensor): The ultrasonic sensor on the left
        ultrasonic_right (IDPDistanceSensor): The ultrasonic sensor on the right
        infrared (IDPDistanceSensor): The IR sensor (long range)
        color_detector (IDPColorDetector): The colour detector, containing two light sensors with red and green filters
    """

    def __init__(self):
        super().__init__()

        self.DEBUG_OBJECTIVE = False  # Change this in driver.py not here

        # Motion properties, derived experimentally, speeds are when drive = 1
        self.max_possible_speed = {"f": 0.4, "r": 4.3}  # THESE MUST BE ACCURATE, else things get  w e i r d
        self.default_max_allowed_speed = {"f": 0.4, "r": 3.3}
        self.max_acc = {"f": 1.2, "r": 7.4}
        MotionCS.max_f_speed = self.default_max_allowed_speed["f"]
        # These are tunable if the robot is slipping or gripping more than expected

        # note the sensors are assumed to be at the centre of the robot, and the robot is assumed symmetrical
        self.color = self.getName()
        if self.color not in ['red', 'green']:
            raise Exception('Name the robot either red or green')
        self.home = [-0.4, 0] if self.color == 'green' else [0.4, 0]

        self.last_bearing = None

        self.arena_length = 2.4

        self.timestep = int(self.getBasicTimeStep())
        self.timestep_actual = self.timestep / 1000  # Webots timestep is in ms
        self.time = 0

        # Devices
        self.gps = IDPGPS('gps', self.timestep)
        self.compass = IDPCompass('compass', self.timestep)
        self.ultrasonic_left = IDPDistanceSensor('ultrasonic_left', self.timestep)
        self.ultrasonic_right = IDPDistanceSensor('ultrasonic_right', self.timestep)
        self.infrared = IDPDistanceSensor('infrared', self.timestep, decreasing=True, min_range=0.15)
        self.motors = IDPMotorController('wheel1', 'wheel2', self)
        self.radio = IDPRadio(self.timestep)
        self.color_detector = IDPColorDetector(self.timestep)
        self.gate = IDPGate('gate')
        self.map = Map(self, self.infrared, self.arena_length, 'map')

        self.distance_sensor_offset = 0.075

        # To store and process detections
        self.targeting_handler = TargetingHandler()
        self.target_cache = TargetCache()
        self.target = None

        # Store internal action queue
        self.action_queue = []

        # Store the function associated with each action
        self.action_functions = {
            "move": self.drive_to_position,
            "face": self.face_bearing,
            "rotate": self.rotate,
            "reverse": self.reverse_to_position,
            "collect": self.collect_block,
            "scan": self.scan,
            "brake": self.brake,
            "hold": self.hold
        }

        # So we can cleanup if we change our action
        self.last_action_type = None
        self.last_action_value = None

        # State for some composite actions
        self.collect_state = IDPRobotState.START_COLLECT
        self.collect_num_tries = 0
        self.stored_time = 0
        self.collect_color_readings = []
        self.collect_far_approach_pos = None

        # Thresholds for finishing actions, speeds determined by holding that quantity for a given time period
        self.hold_time = 1.0  # s
        self.default_target_distance_threshold = 0.05
        self.default_target_bearing_threshold = tau / 360

        # For rotations
        self.rotation_angle = 0
        self.angle_rotated = 0
        self.rotating = False

        # For getting stuck
        self.stuck_in_drive_to_pos_time = 0
        self.angle_rotated_in_drive_to_position = 0

        # Motion control, note: Strongly recommended to use K_d=0 for velocity controllers due to noise in acceleration
        self.pid_f_velocity = PID(1, 0, 0, self.timestep_actual, quantity_name="Forward Velocity",
                                  timer_func=self.getTime)
        self.pid_distance = PID(5, 0, 0, self.timestep_actual, quantity_name="Distance", timer_func=self.getTime)

        def non_lin_controller1(error, cumulative_error, error_change):
            def log_w_sign(x, inner_coefficient):
                return np.log((inner_coefficient * abs(x)) + 1) * np.sign(x)
            return (0.5 * log_w_sign(error, 10)) + (0.0 * cumulative_error) + (0.15 * error_change)

        self.pid_angle = PID(custom_function=non_lin_controller1, time_step=self.timestep_actual,
                             derivative_weight_decay_half_life=0.025, quantity_name="Angle", timer_func=self.getTime,
                             integral_delay_time=1, integral_wind_up_speed=0.5, integral_active_error_band=tau/4,
                             integral_delay_windup_when_in_bounds=True)

        motor_graph_styles = {"distance": 'k-', "angle": 'r-', "forward_speed": 'k--', "rotation_speed": 'r--',
                              "linear_speed": "k:", "angular_velocity": "r:", "left_motor": 'b-', "right_motor": 'y-'}
        self.motion_history = DataRecorder("time", "distance", "angle", "forward_speed", "rotation_speed", "left_motor",
                                           "right_motor", "linear_speed", "angular_velocity", styles=motor_graph_styles)

    def step(self, timestep):
        """A wrapper for the step call that allows us to keep our last bearing and keep track of time. Furthermore,
        tasks that needs to be ran every timestep are also put here.
        """
        self.last_bearing = self.bearing if self.time != 0 else None
        self.time += self.timestep_actual
        returned = super().step(timestep)

        radio_message_data = {
            'position': self.position,
            'bearing': self.bearing,
            'vertices': list(map(list, self.get_bot_vertices())),
            'collected': self.target_cache.prepare_collected_message(),
            'invalid_targets': self.target_cache.prepare_invalid_message()
        }

        # Send 'target': [x, y] if a target is selected
        if self.target:
            radio_message_data['target'] = list(self.target.position)

        # Remove targets already collected by the other robot
        self.target_cache.remove_invalid(self.radio.get_other_bot_invalid_targets())

        # Send our colour detected targets to the other robot
        if self.target_cache.targets:
            radio_message_data['targets_info'] = [[list(t.position), t.classification] for t in self.target_cache.targets
                                                  if not t.sent_to_other_bot]
            for target in self.target_cache.targets:
                target.sent_to_other_bot = True

        self.radio.send_message(radio_message_data)  # also sending 'confirmed': (pos, color) if a block should be collected by the other robot

        self.radio.dispatch_message()  # TODO ideally this should be send at the end of the timestep

        # Add targets the other robot has scanned
        if other_bots_targets := self.radio.get_other_bot_targets_info():
            for target_info in other_bots_targets:
                self.target_cache.add_target(target_info[0], target_info[1])

        # update the map
        for t in self.target_cache.targets:  # plotting target markers
            self.map.plot_coordinate(
                t.position,
                style='o' if t.classification in ['box', f'{self.color}_box'] else 's'
            )
        self.map.update()

        return returned

    def getTime(self):
        """This function is used by PIDs to see what the current robot time is for accurate data recording"""
        return self.time

    def getDevice(self, name: str):
        # here to make sure no device is retrieved this way
        if name in ['gps', 'compass', 'wheel1', 'wheel2', 'ultrasonic_left',
                    'ultrasonic_right', 'infrared', "red_light_sensor", "green_light_sensor"]:
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
    def linear_speed(self) -> float:
        """The current speed of the robot measured by the GPS

        Returns:
            float: Current speed (ms^-1)
        """
        return self.gps.getSpeed()

    @property
    def angular_velocity(self) -> float:
        """The current angular velocity of the robot measured by bearings

        Returns:
            float: Angular velocity, rad/s
        """
        return -self.angle_from_bot_from_bearing(self.last_bearing) / self.timestep_actual \
            if self.last_bearing is not None else 0

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
        return rad + tau if rad < 0 else rad

    def distance_from_bot(self, position: Union[list, np.ndarray]) -> float:
        """The Euclidean distance between the bot and a position

        Args:
            position ([float, float]): Positions co-ordinates, East-North, m

        Returns:
            float: Distance between bot and target in metres
        """
        relative_position = np.asarray(position) - np.asarray(self.position)
        distance = np.hypot(*relative_position)
        return distance

    def angle_from_bot_from_bearing(self, bearing: float) -> float:
        """The clockwise angle from the direction our bot is facing to the bearing in radians

        Args:
            bearing (float): Bearing from north

        Returns:
            float: Angle measured clockwise from direction bot is facing, [-pi, pi]
        """
        angle = bearing - self.bearing

        # Need to adjust if outside [-pi,pi]
        if angle > tau / 2:
            angle -= tau
        elif angle < -(tau / 2):
            angle += tau

        return angle

    def angle_from_bot_from_position(self, position: Union[list, np.ndarray]) -> float:
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

        center_to_topleft = np.array([-0.1, 0.225])
        center_to_topright = np.array([0.175, 0.225])
        center_to_bottomright = np.array([0.175, -0.12])
        center_to_bottomleft = np.array([-0.1, -0.12])

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

    def get_min_distance_point_to_bot(self, position: Union[list, np.ndarray]) -> float:
        """Get the minimum distance between a point and the bounding box of the robot

        Args:
            position (list, np.ndarray): The coordinate of the point

        Returns:
            float: The minimum distance
        """
        return get_min_distance_point_rectangle(get_rectangle_sides(self.get_bot_vertices()), position)

    def get_min_distance_bot_to_bot(self, other_bot_vertices: list) -> float:
        """Get the minimum distances between two robots

        Args:
            other_bot_vertices ([np.ndarray]): List of vertices of the other robot, needs to be specified in
            a clockwise order

        Returns:
            float: The minimum distance between the two robots
        """
        return get_min_distance_rectangles(self.get_bot_vertices(), other_bot_vertices)

    def plot_motion_history(self):
        self.motion_history.plot("time", title="Robot motor graph")

    def plot_all_graphs(self):
        fire_and_forget(self.plot_motion_history)
        fire_and_forget(self.pid_f_velocity.plot_history)
        fire_and_forget(self.pid_distance.plot_history)
        fire_and_forget(self.pid_angle.plot_history)

    def update_motion_history(self, **kwargs):
        self.motion_history.update(left_motor=self.motors.velocities[0], right_motor=self.motors.velocities[1],
                                   **kwargs)

    def reset_action_variables(self):
        """Cleanup method to be called when the current action changes. If executing bot commands manually
        (i.e. robot.drive_to_position), call this first.
        """
        self.rotation_angle = 0
        self.angle_rotated = 0
        self.rotating = False
        self.last_action_type = None
        self.last_action_value = None
        self.collect_state = IDPRobotState.START_COLLECT
        self.stored_time = 0
        self.stuck_in_drive_to_pos_time = 0
        self.angle_rotated_in_drive_to_position = 0

    def passive_collision_avoidance(self, target_pos, angle):
        # Build up a list of obstructions to avoid which includes their type
        other_bot_pos = self.radio.get_other_bot_position()
        known_block_positions = [np.array(getattr(target, 'position')) for target in self.target_cache.targets
                                 if (self.target is None or target != self.target)]
        obstructions = [{'type': 'block', 'position': pos} for pos in known_block_positions]\
            + ([{'type': 'bot', 'position': np.array(other_bot_pos)}] if other_bot_pos is not None else [])

        # Some tunable parameters
        min_approach_dist = {'block': 0.2, 'bot': 0.45}
        avoidance_bandwidth = 0.2

        # Here we consider if two obstructions are close enough to constitute treatment as one large obstructions
        # and if so, we dynamically generate a new center and approaches for it
        new_obstructions = 0
        print_if_debug("\n", debug_flag=DEBUG_OBSTRUCTIONS)
        for o in obstructions:
            print_if_debug(o, debug_flag=DEBUG_OBSTRUCTIONS)

        def combine_obstructions(obstruction_list):
            nonlocal new_obstructions
            combined_obstructions = []
            new_list = []

            for i, x in enumerate(obstruction_list):
                if i not in combined_obstructions:
                    for j, y in enumerate(obstruction_list):
                        if i != j and j not in combined_obstructions:

                            dist = np.linalg.norm(x['position'] - y['position'])
                            min_approaches = [min_approach_dist[x['type']], min_approach_dist[y['type']]]

                            if dist < min_approaches[0] + min_approaches[1]:

                                # Find bounding circle
                                dir_vec = (x['position'] - y['position']) / dist
                                center = (x['position'] + y['position'] - ((min_approaches[1] - min_approaches[0]) * dir_vec)) / 2
                                radius = (min_approaches[0] + min_approaches[1] + dist) / 2

                                # In event one of our old circles was inside the other
                                if radius < min_approaches[0]:
                                    radius = min_approaches[0]
                                    center = x['position']
                                if radius < min_approaches[1]:
                                    radius = min_approaches[1]
                                    center = y['position']

                                # Add to new list
                                new_type = f'custom{new_obstructions}'
                                new_obstructions += 1
                                new_list.append({'type': new_type, 'position': center})
                                min_approach_dist[new_type] = radius
                                combined_obstructions.extend([i, j])
                                break
                    else:
                        new_list.append(x)

            if new_list != obstruction_list:
                return combine_obstructions(new_list)
            else:
                return new_list

        obstructions = combine_obstructions(obstructions)

        print_if_debug("", debug_flag=DEBUG_OBSTRUCTIONS)
        for o in obstructions:
            print_if_debug(o, min_approach_dist[o['type']], debug_flag=DEBUG_OBSTRUCTIONS)


        # Iterate through and build up a list of avoidance angles and angle fractions
        angles_and_fractions = []
        for obstruction in obstructions:

            # Check the obstruction is not at our target position, if we are just get as close as possible
            distance_to_obstruction = self.distance_from_bot(obstruction['position'])
            distance_from_target_to_obstruction = np.linalg.norm(obstruction['position'] - target_pos)
            if distance_from_target_to_obstruction < min_approach_dist[obstruction['type']] and\
                    distance_to_obstruction < min_approach_dist[obstruction['type']]\
                    + self.default_target_distance_threshold:
                print_if_debug(f"{self.color}, collision: Obstruction is where we want to go and we are close,\
stopping here", debug_flag=DEBUG_COLLISIONS)
                return None

            # Calculate the angle we would need to turn to (i.e. have the PID minimise) to avoid the obstruction
            angle_to_obstruction = self.angle_from_bot_from_position(obstruction['position'])
            angle_to_avoid_obstruction = angle_to_obstruction + (np.sign(angle - angle_to_obstruction) * tau/4)

            # Calculate the angle fraction for this angle
            angle_fraction = ((min_approach_dist[obstruction['type']] + avoidance_bandwidth) - distance_to_obstruction) / avoidance_bandwidth
            angle_fraction = min(max(angle_fraction, 0), 2)
            angles_and_fractions.append([angle_to_avoid_obstruction, angle_fraction])
        print_if_debug(angles_and_fractions, debug_flag=DEBUG_OBSTRUCTIONS)
        print_if_debug(angle, debug_flag=DEBUG_OBSTRUCTIONS)

        # Normalise so our sum of the fractions is equal to the current highest value
        fraction_total = sum(x[1] for x in angles_and_fractions)
        if fraction_total != 0:
            max_fraction = max(x[1] for x in angles_and_fractions)
            angles_and_fractions = [[x[0], x[1] * max_fraction / fraction_total] for x in angles_and_fractions]

            # Determine our final angle to turn to, to hopefully avoid obstructions whilst reaching our target
            angle = sum([x[0] * x[1] for x in angles_and_fractions])\
                + ((1 - sum(x[1] for x in angles_and_fractions)) * angle)
        print_if_debug(angle, debug_flag=DEBUG_OBSTRUCTIONS)

        return angle

    def drive_to_position(self, target_pos: Union[list, np.ndarray], max_forward_speed=None, max_rotation_rate=None,
                          reverse=False, passive_collision_avoidance=True, accuracy_threshold=None) -> bool:
        """Go to a position

        Args:
            target_pos ([float, float]): The East-North co-ords of the target position
            max_forward_speed (float): Maximum speed to travel at m/s
            max_rotation_rate (float): Maximum rate to rotate at, rad/s
            reverse (bool): Whether to reverse there
            passive_collision_avoidance (bool): Whether to try and avoid the other bot and known blocks
            accuracy_threshold (float): Threshold determining whether the target coordinate is reached
        Returns:
            bool: If we are at our target
        """
        accuracy_threshold = self.default_target_distance_threshold \
            if accuracy_threshold is None else accuracy_threshold

        target_pos = np.asarray(target_pos)

        max_rotation_rate = self.default_max_allowed_speed["r"] if max_rotation_rate is None else max_rotation_rate
        max_rotation_drive = max_rotation_rate / self.max_possible_speed["r"]

        max_forward_speed = self.default_max_allowed_speed["f"] if max_forward_speed is None else max_forward_speed
        max_forward_drive = max_forward_speed / self.max_possible_speed["f"]

        distance = self.distance_from_bot(target_pos)
        angle = self.angle_from_bot_from_position(target_pos)

        if distance <= accuracy_threshold and self.linear_speed <= accuracy_threshold / self.hold_time:
            self.angle_rotated_in_drive_to_position = 0
            return True

        # In-case we get stuck at wall
        if self.linear_speed <= accuracy_threshold / self.hold_time:
            if self.stuck_in_drive_to_pos_time >= 1.5:
                self.stuck_in_drive_to_pos_time = 0
                print_if_debug(f"{self.color}, stuck: Not moved in 1.5s, stopping move", debug_flag=DEBUG_STUCK)
                self.angle_rotated_in_drive_to_position = 0
                return True
            else:
                self.stuck_in_drive_to_pos_time += self.timestep_actual
                
        # In-case we get stuck in a spin
        if distance <= 4 * accuracy_threshold:
            self.angle_rotated_in_drive_to_position += self.angular_velocity * self.timestep_actual
            if abs(self.angle_rotated_in_drive_to_position) >= tau * 1.5:
                print_if_debug(f"{self.color}, stuck: Done 1.5 full rotations near target, stopping move",
                               debug_flag=DEBUG_STUCK)
                self.angle_rotated_in_drive_to_position = 0
                return True

        # If close to the other bot, turn to avoid it.
        if passive_collision_avoidance:
            angle = self.passive_collision_avoidance(target_pos, angle)
            if angle is None:  # target_pos is blocked by an obstruction and we are as close as possible
                return True

        # If we're reversing we change the angle so it mimics the bot facing the opposite way
        # When we apply the wheel velocities we negative them and voila we tricked the bot into reversing
        angle = (np.sign(angle) * (tau / 2)) - angle if reverse else angle

        forward_speed = min(MotionCS.linear_dual_pid(distance=distance, distance_pid=self.pid_distance, angle=angle,
                                                     forward_speed=self.linear_speed,
                                                     forward_speed_pid=self.pid_f_velocity), max_forward_drive)

        r_speed = self.pid_angle.step(angle)
        rotation_speed = sorted([r_speed, np.sign(r_speed) * max_rotation_drive], key=lambda x: abs(x))[0]

        raw_velocities = MotionCS.combine_and_scale(forward_speed, rotation_speed)
        self.motors.velocities = raw_velocities if not reverse else -raw_velocities
        self.update_motion_history(time=self.time, distance=distance, angle=angle, forward_speed=forward_speed,
                                   rotation_speed=rotation_speed, linear_speed=self.linear_speed,
                                   angular_velocity=self.angular_velocity)
        return False

    def reverse_to_position(self, target_pos: Union[list, np.ndarray]) -> bool:
        """Go to a position in reverse

        Args:
            target_pos ([float, float]): The East-North co-ords of the target position
        Returns:
            bool: If we are at our target
        """
        return self.drive_to_position(target_pos, reverse=True, passive_collision_avoidance=False)

    def rotate(self, angle: float, max_rotation_rate=None, accuracy_threshold=None) -> bool:
        """Rotate the bot a fixed angle at a fixed rate of rotation

        Args:
            angle (float): Angle to rotate in radians, positive is clockwise
            max_rotation_rate (float): Maximum rate to rotate at, rad/s
            accuracy_threshold (float): Threshold determining whether the target angle is reached
        Returns:
            bool: If we completed rotation
        """
        accuracy_threshold = self.default_target_bearing_threshold \
            if accuracy_threshold is None else accuracy_threshold

        max_rotation_rate = self.default_max_allowed_speed["r"] if max_rotation_rate is None else max_rotation_rate
        max_rotation_drive = max_rotation_rate / self.max_possible_speed["r"]

        # Check if it's a new rotation and update how far we've rotated
        self.rotation_angle = self.rotation_angle if self.rotating else angle
        self.angle_rotated += self.angular_velocity * self.timestep_actual
        self.rotating = True

        # Check if we're done
        angle_difference = self.rotation_angle - self.angle_rotated
        if abs(angle_difference) <= accuracy_threshold and \
                abs(self.angular_velocity) <= accuracy_threshold / self.hold_time:
            self.rotation_angle = 0
            self.angle_rotated = 0
            self.rotating = False
            return True

        r_speed = self.pid_angle.step(angle_difference)
        rotation_speed = sorted([r_speed, np.sign(r_speed) * max_rotation_drive], key=lambda x: abs(x))[0]
        self.motors.velocities = MotionCS.combine_and_scale(0, rotation_speed)
        self.update_motion_history(time=self.time, angle=angle_difference, rotation_speed=rotation_speed,
                                   linear_speed=self.linear_speed, angular_velocity=self.angular_velocity)
        return False

    def face_bearing(self, target_bearing: float) -> bool:
        """Face a given bearing

        Args:
            target_bearing (float): Desired bearing of our robot
        Returns:
            bool: If we are at our target
        """
        return self.rotate(self.angle_from_bot_from_bearing(target_bearing))

    def brake(self):
        self.motors.velocities = np.zeros(2)
        self.update_motion_history(time=self.time, linear_speed=self.linear_speed,
                                   angular_velocity=self.angular_velocity)
        return abs(self.linear_speed) <= self.default_target_distance_threshold / self.hold_time \
            and abs(self.angular_velocity) <= self.default_target_bearing_threshold / self.hold_time

    def hold(self, time=None):
        # Store when we want hold to end, we do this instead of storing current time because current time might be 0
        if time is not None:
            if self.stored_time == 0:
                self.stored_time = self.time + time
            if self.time >= self.stored_time:
                return True
        self.brake()
        return False

    def collect_block(self) -> bool:
        """Collect block at the best position possible

        Returns:
            bool: If the collect action is completed
        """
        if any((
                not self.target,  # no target selected
                all((
                        self.collect_state in [IDPRobotState.START_COLLECT, IDPRobotState.DRIVING_TO_TARGET,
                                               IDPRobotState.APPROACHING_TARGET_FROM_CENTER,
                                               IDPRobotState.ROTATING_TO_FACE_TARGET_AFTER_APPROACH,
                                               IDPRobotState.ROTATING_TO_FACE_TARGET_FOR_DETECT,
                                               IDPRobotState.DETECTING_COLOUR],  # on the way to target
                        not self.check_target_valid(self.target),  # collision if continue to target
                        other_bot_pos := self.radio.get_other_bot_position(),  # have position data of the other robot
                        abs(self.angle_from_bot_from_position(other_bot_pos)) < np.pi / 2
                        # the other robot is in front of us
                ))
        )):
            if curr_best_target := self.get_best_target():
                self.target = curr_best_target  # update target to the best available at this time
                print_if_debug(f"{self.color}, objective: Old target not best, now going to collect block at \
{self.target.position}",
                               debug_flag=self.DEBUG_OBJECTIVE)
                self.collect_state = IDPRobotState.START_COLLECT
            else:  # no other target can be selected either, return to scan
                print_if_debug(f"{self.color}, collect: No target can be selected", debug_flag=DEBUG_COLLECT)
                self.brake()
                return True

        # PLEASE NOTE! Since the distance accuracy is 0.02, the bot will stop ~0.02 distance from its goal
        # If the goal is within 0.02 it won't move at all as it's 'already there'
        colour_detect_distance_start = 0.25
        colour_detect_distance_end = 0.15
        max_angle_to_block = 0.12

        block_nearby_threshold = 0.5
        rotate_angle = -tau / 2.5
        rotate_angle_when_block_nearby = -tau / 5

        gate_time = 0.5
        reverse_distance = 0.2
        collect_rotation_rate = 2.0

        # Ifs not elifs means we don't waste timesteps if the state changes

        if self.collect_state == IDPRobotState.START_COLLECT:
            self.collect_far_approach_pos = [0.75 * np.sign(x) if abs(x) > 1.0 else x for x in self.target.position]
            if any(x != y for x, y in zip(self.collect_far_approach_pos, self.target.position)):
                print_if_debug(f"{self.color}, collect: Block is near edge, driving to {self.collect_far_approach_pos}",
                               debug_flag=DEBUG_COLLECT)
                self.collect_state = IDPRobotState.APPROACHING_TARGET_FROM_CENTER
            else:
                self.collect_far_approach_pos = None
                self.collect_state = IDPRobotState.DRIVING_TO_TARGET

        if self.collect_state == IDPRobotState.APPROACHING_TARGET_FROM_CENTER:  # Approach wide if at edge
            if self.drive_to_position(self.collect_far_approach_pos):  # Use passive collision avoidance
                self.collect_far_approach_pos = None
                print_if_debug(f"{self.color}, collect: At approach, turning", debug_flag=DEBUG_COLLECT)
                self.collect_state = IDPRobotState.ROTATING_TO_FACE_TARGET_AFTER_APPROACH

        if self.collect_state == IDPRobotState.ROTATING_TO_FACE_TARGET_AFTER_APPROACH:
            angle_to_block = self.angle_from_bot_from_position(self.target.position)
            if self.rotate(angle_to_block, accuracy_threshold=max_angle_to_block):
                print_if_debug(f"{self.color}, collect: Facing target at approach, getting closer",
                               debug_flag=DEBUG_COLLECT)
                self.collect_state = IDPRobotState.DRIVING_TO_TARGET

        if self.collect_state == IDPRobotState.DRIVING_TO_TARGET:  # driving to target
            if self.drive_to_position(self.target.position) \
                    or self.distance_from_bot(self.target.position) <= colour_detect_distance_start:
                print_if_debug(f"{self.color}, collect: At target, rotating", debug_flag=DEBUG_COLLECT)
                self.collect_state = IDPRobotState.ROTATING_TO_FACE_TARGET_FOR_DETECT

        if self.collect_state == IDPRobotState.ROTATING_TO_FACE_TARGET_FOR_DETECT:
            angle_to_block = self.angle_from_bot_from_position(self.target.position)
            if self.rotate(angle_to_block, accuracy_threshold=max_angle_to_block):
                print_if_debug(f"{self.color}, collect: Facing target, detecting color", debug_flag=DEBUG_COLLECT)
                self.collect_state = IDPRobotState.DETECTING_COLOUR

        if self.collect_state == IDPRobotState.DETECTING_COLOUR:
            # Log current detection
            detection = self.color_detector.get_color()
            if detection in ["red", "green"]:
                self.collect_color_readings.append(detection)

            # Done with move
            if self.drive_to_position(self.target.position, max_forward_speed=0.1, passive_collision_avoidance=False) \
                    or self.distance_from_bot(self.target.position) <= colour_detect_distance_end:

                # Other bot has already scanned it, putting this here and not earlier so bot still drives close to block
                if self.target.classification == f'{self.color}_box':
                    print_if_debug(f"{self.color}, collect: Apparently ours, collecting",
                                   debug_flag=DEBUG_COLLECT)
                    self.collect_state = IDPRobotState.BRAKING

                else:
                    # Get the most common color occurrence
                    red_count = self.collect_color_readings.count("red")
                    green_count = self.collect_color_readings.count("green")
                    self.collect_color_readings = []
                    if (red_count != 0 or green_count != 0) and red_count != green_count:
                        if red_count > green_count:
                            color = "red"
                        else:
                            color = "green"
                        print(f"Block colour: {color}")
                        self.target.classification = f"{color}_box"

                        if color == self.color:
                            print_if_debug(f"{self.color}, collect: Color match, collecting",
                                           debug_flag=DEBUG_COLLECT)
                            self.collect_state = IDPRobotState.BRAKING
                        else:
                            self.action_queue.insert(1, ("reverse", list(self.get_bot_front(-reverse_distance))))
                            self.target.sent_to_other_bot = False  # Need to resend to update info
                            self.target = None
                            print_if_debug(f"{self.color}, collect: Color opposite, reversing",
                                           debug_flag=DEBUG_COLLECT)
                            return True

                    # Not able to detect the colour, probably because the position is not accurate or we were going to
                    # collide with something
                    else:
                        self.action_queue.insert(1, ("reverse", list(self.get_bot_front(-reverse_distance))))
                        self.action_queue.insert(2, "scan")  # rescan to check if there is actually a target there
                        # pop the current target off for now, the new scan will give better position
                        if self.collect_num_tries > 1:  # this block is probably flipped over
                            self.target.classification = 'flipped'
                            self.collect_num_tries = 0  # reset the counter
                            print_if_debug(f"{self.color}, collect: Color unknown, I think this is flipped",
                                           debug_flag=DEBUG_COLLECT)
                        else:
                            self.target_cache.disappeared.append(self.target)
                            self.target_cache.remove_target(self.target)
                            self.collect_num_tries += 1
                            print_if_debug(f"{self.color}, collect: Color unknown, I'll try again later",
                                           debug_flag=DEBUG_COLLECT)
                        self.target = None
                        return True

        if self.collect_state == IDPRobotState.BRAKING:
            if self.brake():
                self.stored_time = self.time
                self.collect_state = IDPRobotState.GATE_OPENING

        if self.collect_state == IDPRobotState.GATE_OPENING:
            self.gate.open()
            if self.time - self.stored_time >= gate_time:
                print_if_debug(f"{self.color}, collect: Gate open, checking rotation", debug_flag=DEBUG_COLLECT)
                self.collect_state = IDPRobotState.CHECKING_ROTATE

        if self.collect_state == IDPRobotState.CHECKING_ROTATE:
            # We need to check if there are any blocks nearby our target, if there are we need to rotate less to avoid
            # mishaps. If this changes during the motion rotate will ignore that
            for block in self.target_cache.targets:
                if block != self.target:  # Don't want to compare our current block to our current block lol
                    if self.target.is_near(block.position, threshold=block_nearby_threshold):
                        rotate_angle = rotate_angle_when_block_nearby
                        self.action_queue.insert(1, ("rotate", -rotate_angle))
                        self.action_queue.insert(2, ("reverse", list(self.get_bot_front(-reverse_distance))))
                        print_if_debug(f"{self.color}, collect: Block nearby, rotating less and recovering",
                                       debug_flag=DEBUG_COLLECT)
                        break
            self.collect_state = IDPRobotState.ROTATING_TO_COLLECT

        if self.collect_state == IDPRobotState.ROTATING_TO_COLLECT:
            if self.rotate(rotate_angle, max_rotation_rate=collect_rotation_rate):
                self.stored_time = self.time
                print_if_debug(f"{self.color}, collect: Collected, closing gate", debug_flag=DEBUG_COLLECT)
                self.collect_state = IDPRobotState.GATE_CLOSING

        if self.collect_state == IDPRobotState.GATE_CLOSING:
            self.gate.close()
            if self.time - self.stored_time >= gate_time:
                self.target_cache.remove_target(self.target)
                self.target_cache.collected.append(self.target)
                self.target = None
                print_if_debug(f"{self.color}, collect: Gate closed, moving on", debug_flag=DEBUG_COLLECT)
                return True

        return False

    def scan(self) -> bool:
        """Rotate 360 degrees to scan for blocks

        Returns:
            bool: Whether the scan is completed
        """
        if not self.targeting_handler.relocating:
            complete = self.rotate(tau, max_rotation_rate=2.0)

            distance = self.infrared.getValue()
            d_min, d_max = self.infrared.getBounds()
            bound = (d_max - d_min) * 2

            if abs(self.get_sensor_distance_to_wall() - distance) > bound * 1.5 \
                    and abs(self.infrared.max_range - distance) > bound:
                pos = self.get_bot_front(distance + 0.0125)  # add a bit to get the centre of the block
                other_bot_pos = self.radio.get_other_bot_position()
                if other_bot_pos is None or not Target.check_near(pos, other_bot_pos, 0.3):  # not the other robot
                    self.targeting_handler.positions.append(pos)
                    # self.targeting_handler.bounds.append(bound)

            if complete:
                for target_pos in self.targeting_handler.get_targets(self.position):
                    self.target_cache.add_target(target_pos)

                print_if_debug(f"{self.color}, targets: {self.target_cache.targets}", debug_flag=DEBUG_TARGETS)

                if self.get_best_target() is None:
                    self.targeting_handler.next_scan_position = self.targeting_handler.get_fallback_scan_position(
                        self.infrared.max_range)
                    self.targeting_handler.relocating = True
                    print_if_debug(f"{self.color}, scan: No targets found, relocating to \
{self.targeting_handler.next_scan_position}", debug_flag=DEBUG_SCAN)

        else:
            complete = self.drive_to_position(self.targeting_handler.next_scan_position)
            self.targeting_handler.relocating = not complete

        return complete

    def do(self, *args):
        """Small wrapper to make telling robot to do something a little cleaner"""
        self.action_queue = [args]

    def execute_next_action(self) -> bool:
        """Execute the next action in self.action_queue

        When each action is completed it's removed from the list. Using list mutability this allows us to alter / check
        the action list elsewhere in the code to see the bots progress and also change its objectives.
        If people have better ideas on how to do this I'm all ears.
        Execute_action was only ever supposed to be about motion

        Returns:
            bool: Whether action list is completed or not
        """

        # Check if action list is empty i.e. 'complete'
        if len(self.action_queue) == 0:
            self.motors.velocities = np.zeros(2)
            return True

        # Execute action
        action_item = ensure_list_or_tuple(self.action_queue[0])
        action_type = action_item[0]
        action_value = action_item[1:]

        # Check action is valid
        if action_type not in self.action_functions.keys():
            raise Exception(
                f"Action {action_type} is not a valid action, valid actions: {', '.join(self.action_functions.keys())}")

        # If we are changing our action we might need to reset some rotation stuff
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
            print_if_debug(f"\n{self.color}, actions:\nCompleted action {self.action_queue[0]}",
                           debug_flag=DEBUG_ACTIONS)
            del self.action_queue[0]
            print_if_debug(f"Remaining actions:", debug_flag=DEBUG_ACTIONS)

            # Check if action list is now empty
            if len(self.action_queue) == 0:
                print_if_debug("None", debug_flag=DEBUG_ACTIONS)
                self.motors.velocities = np.zeros(2)
                return True

            print_if_debug('\n'.join(str(x) for x in self.action_queue), debug_flag=DEBUG_ACTIONS)

        return False

    def check_target_valid(self, target: Union[Target, None]) -> bool:
        good_path = target.classification in ['box', f'{self.color}_box'] and \
            not self.target_cache.check_target_path_blocked(
                target.position,
                self.position,
                self.radio.get_other_bot_position(),
                self.radio.get_other_bot_vertices()
            ) if target else False

        # Check we don't have the same target as other bot, arbitrarily give green priority
        same_as_other_bot = False
        if self.color == "red" and target is not None and self.radio.get_other_bot_target() is not None:
            same_as_other_bot = target.is_near(self.radio.get_other_bot_target())

        return good_path and not same_as_other_bot

    def filter_targets(self, targets: list) -> list:
        """Filter a given list of targets, returns targets the robot can drive to without hitting other targets or
        the other robot on the way.

        Args:
            targets ([Target]): List of targets

        Returns:
            [Target]: Filtered list of targets
        """
        return list(filter(
            self.check_target_valid,
            targets
        ))

    def get_best_target(self) -> Union[Target, None]:
        """Decide on a new target block for robot, which is the closest valid target on the way to which
        there won't involve any collision

        Returns:
            [float, float]: Targets co-ordinates, East-North, m
        """
        potential_targets_sorted = self.target_cache.get_targets(
            valid_classes=['box', f'{self.color}_box'],
            key=lambda t: self.distance_from_bot(t.position)
        )

        valid_targets = self.filter_targets(potential_targets_sorted)
        return valid_targets[0] if valid_targets else None

    def get_imminent_collision(self) -> Union[tuple, None]:
        if self.radio.get_other_bot_position() is None or self.radio.get_other_bot_vertices() is None:
            return None

        zone_length = 0.5
        front_rectangle = list(map(
            self.coordtransform_bot_cartesian_to_world,
            [
                np.array([-0.2, 0.225 + zone_length]),  # topleft
                np.array([0.3, 0.225 + zone_length]),
                np.array([-0.2, -0.1]),
                np.array([0.3, -0.1])  # bottomright
            ]
        ))

        sorted_by_dist = sorted(
            map(
                lambda tp: (tp, self.get_min_distance_point_to_bot(np.asarray(tp))),
                chain(
                    map(
                        attrgetter('position'),
                        filter(
                            lambda t: t.classification in ['box', 'green_box', 'red_box', 'flipped'],
                            self.target_cache.targets
                        )
                    ),
                    self.radio.get_other_bot_vertices()
                )
            ),
            key=lambda x: x[1]
        )

        for pos_d in sorted_by_dist:
            if point_in_rectangle(front_rectangle, pos_d[0]):
                return pos_d

        return None
