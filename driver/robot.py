# Copyright (C) 2021 Weixuan Zhang, Tim Clifford, Jason Brown
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
from modules.utils import print_if_debug, ensure_list_or_tuple
from modules.geometry import rotate_vector, get_rectangle_sides, get_min_distance_point_rectangle, \
    get_min_distance_rectangles, point_in_rectangle
from modules.mapping import Map
from modules.pid import PID, DataRecorder
from modules.targeting import TargetingHandler, Target, TargetCache

DEBUG = False
tau = np.pi * 2


class IDPRobotState(Enum):
    DRIVING_TO_TARGET                   = 0
    ROTATE_TO_FACE_TARGET               = 1
    FACING_TARGET                       = 2
    DETECTING_COLOUR                    = 3
    GET_TO_COLLECT_DISTANCE_FROM_BLOCK  = 4
    CORRECT_COLOUR                      = 5
    COLLECTING_TARGET                   = 6
    TARGET_COLLECTED                    = 7


class IDPRobot(Robot):
    """Class representing the robot

    Attributes:
        compass (IDPCompass): The compass
        gps (IDPGPS): The GPS
        motors (IDPMotorController): The two motors
        target_bearing_threshold (float): Threshold determining whether the target bearing is reached
        target_distance_threshold (float): Threshold determining whether the target coordinate is reached
        timestep (float): Time step of the current world
        ultrasonic_left (IDPDistanceSensor): The ultrasonic sensor on the left
        ultrasonic_right (IDPDistanceSensor): The ultrasonic sensor on the right
        infrared (IDPDistanceSensor): The IR sensor (long range)
        color_detector (IDPColorDetector): The colour detector, containing two light sensors with red and green filters
    """

    def __init__(self):
        super().__init__()

        # Motion properties, derived experimentally, speeds are when drive = 1
        self.max_possible_speed = {"f": 1.0, "r": 11.2}  # THESE MUST BE ACCURATE, else things get  w e i r d
        self.default_max_allowed_speed = {"f": 1.0, "r": 5.0}
        self.max_acc = {"f": 5.0, "r": 40.0}
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
        self.collect_state = IDPRobotState.DRIVING_TO_TARGET
        self.collect_num_tries = 0
        self.stored_time = 0

        # Thresholds for finishing actions, speeds determined by holding that quantity for a given time period
        hold_time = 0.5  # s
        self.target_distance_threshold = 0.02
        self.linear_speed_threshold = self.target_distance_threshold / hold_time
        self.target_bearing_threshold = tau / 360
        self.angular_speed_threshold = self.target_bearing_threshold / hold_time

        # For rotations
        self.rotation_angle = 0
        self.angle_rotated = 0
        self.rotating = False

        # For getting stuck
        self.stuck_steps = 0

        # Motion control, note: Strongly recommended to use K_d=0 for velocity controllers due to noise in acceleration
        self.pid_f_velocity = PID(0.1, 0, 0, self.timestep_actual, quantity_name="Forward Velocity",
                                  timer_func=self.getTime)
        self.pid_distance = PID(2, 0, 0, self.timestep_actual, quantity_name="Distance", timer_func=self.getTime)
        self.pid_angle = PID(2.5, 0.0, 0.10, self.timestep_actual, derivative_weight_decay_half_life=0.05,
                             quantity_name="Angle", timer_func=self.getTime)

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

        self.radio.send_message({
            'position': self.position,
            'bearing': self.bearing,
            'vertices': list(map(list, self.get_bot_vertices())),
            'collected': self.target_cache.prepare_collected_message()
        })  # also sending 'confirmed': (pos, color) if a block should be collected by the other robot

        # send 'target': [x, y] if a target is selected
        if self.target:
            self.radio.send_message({'target': list(self.target.position)})

        self.radio.dispatch_message()  # TODO ideally this should be send at the end of the timestep

        # remove targets already collected by the other robot
        self.target_cache.remove_collected_by_other(self.radio.get_other_bot_collected())

        # add target confirmed by the other robot
        if (confirmed_pos_color := self.radio.get_message().get('confirmed')) \
                and confirmed_pos_color in ['red', 'green']:
            self.target_cache.add_target(confirmed_pos_color[0], f'{confirmed_pos_color[1]}_box')

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

    def angle_from_bot_from_position(self, position: list) -> float:
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
        self.collect_state = IDPRobotState.DRIVING_TO_TARGET
        self.stored_time = 0

    def drive_to_position(self, target_pos: Union[list, np.ndarray], max_forward_speed=None, max_rotation_rate=None,
                          reverse=False, passive_collision_avoidance=True) -> bool:
        """Go to a position

        Args:
            target_pos ([float, float]): The East-North co-ords of the target position
            max_forward_speed (float): Maximum speed to travel at m/s
            max_rotation_rate (float): Maximum rate to rotate at, rad/s
            reverse (bool): Whether to reverse there
            passive_collision_avoidance (bool): Whether to employ ultrasonic collision avoidance
        Returns:
            bool: If we are at our target
        """
        max_rotation_rate = self.default_max_allowed_speed["r"] if max_rotation_rate is None else max_rotation_rate
        max_rotation_drive = max_rotation_rate / self.max_possible_speed["r"]

        max_forward_speed = self.default_max_allowed_speed["f"] if max_forward_speed is None else max_forward_speed
        max_forward_drive = max_forward_speed / self.max_possible_speed["f"]

        # Passive collision avoidance - turn away towards center if path is block
        if passive_collision_avoidance and (blockage_pos_d := self.get_imminent_collision()) is not None:
            print_if_debug(f'robot gonna collide {blockage_pos_d}', debug_flag=DEBUG)
            if self.distance_from_bot(target_pos) > 0.2:  # prevent stuck in rotation
                collision_avoidance_direction = -np.sign(self.angle_from_bot_from_position(blockage_pos_d[0]))

                target_relative_to_bot = np.array([0.2 * collision_avoidance_direction, min(1 / blockage_pos_d[1], 15)])
                target_pos = self.coordtransform_bot_cartesian_to_world(target_relative_to_bot)

        distance = self.distance_from_bot(target_pos)
        angle = self.angle_from_bot_from_position(target_pos)

        if distance <= self.target_distance_threshold and self.linear_speed <= self.linear_speed_threshold:
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

    def rotate(self, angle: float, max_rotation_rate=None) -> bool:
        """Rotate the bot a fixed angle at a fixed rate of rotation

        Args:
            angle (float): Angle to rotate in radians, positive is clockwise
            max_rotation_rate (float): Maximum rate to rotate at, rad/s
        Returns:
            bool: If we completed rotation
        """
        max_rotation_rate = self.default_max_allowed_speed["r"] if max_rotation_rate is None else max_rotation_rate
        max_rotation_drive = max_rotation_rate / self.max_possible_speed["r"]

        # Check if it's a new rotation and update how far we've rotated
        self.rotation_angle = self.rotation_angle if self.rotating else angle
        self.angle_rotated += self.angular_velocity * self.timestep_actual
        self.rotating = True

        # Check if we're done
        angle_difference = self.rotation_angle - self.angle_rotated
        if abs(angle_difference) <= self.target_bearing_threshold and \
                abs(self.angular_velocity) <= self.angular_speed_threshold:
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
        return abs(self.linear_speed) <= self.linear_speed_threshold \
               and abs(self.angular_velocity) <= self.angular_speed_threshold

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
                        self.collect_state == IDPRobotState.DRIVING_TO_TARGET,  # on the way to target
                        not self.check_target_valid(self.target),  # collision if continue to target
                        other_bot_pos := self.radio.get_other_bot_position(),  # have position data of the other robot
                        abs(self.angle_from_bot_from_position(other_bot_pos)) < np.pi / 2
                        # the other robot is in front of us
                ))
        )):
            if curr_best_target := self.get_best_target():
                self.target = curr_best_target  # update target to the best available at this time
            else:  # no other target can be selected either, return to scan
                self.brake()
                return True

        distance_from_block_for_colour_detect = 0.14
        distance_from_block_for_collect = 0.15
        max_angle_to_block = 0.1
        rotate_angle = -tau / 2.5
        gate_time = 0.5
        reverse_distance = 0.2
        collect_rotation_rate = 3.0

        # Ifs not elifs means we don't waste timesteps if the state changes

        if self.collect_state == IDPRobotState.DRIVING_TO_TARGET:  # driving to target
            if self.distance_from_bot(self.target.position) - distance_from_block_for_colour_detect >= 0:
                self.drive_to_position(self.target.position, passive_collision_avoidance=False)
            else:
                self.collect_state = IDPRobotState.ROTATE_TO_FACE_TARGET

        if self.collect_state == IDPRobotState.ROTATE_TO_FACE_TARGET:
            angle_to_block = self.angle_from_bot_from_position(self.target.position)
            if abs(angle_to_block) > max_angle_to_block:
                self.rotate(angle_to_block)
            else:
                self.stored_time = self.time
                self.collect_state = IDPRobotState.FACING_TARGET

        if self.collect_state == IDPRobotState.FACING_TARGET:
            if self.brake():
                self.collect_state = IDPRobotState.DETECTING_COLOUR

        if self.collect_state == IDPRobotState.DETECTING_COLOUR:
            color = self.color_detector.get_color()
            print(f"Block colour: {color}")
            if color in ["red", "green"]:
                self.target.classification = f"{color}_box"
                if color == self.color:
                    self.collect_state = IDPRobotState.GET_TO_COLLECT_DISTANCE_FROM_BLOCK
                else:
                    self.action_queue.insert(1, ("reverse", list(self.get_bot_front(-reverse_distance))))
                    # should be collected by the other robot, send info to it
                    self.radio.send_message({'confirmed': (list(self.target.position), color)})
                    self.target = None
                    return True
            elif (other_bot_collected := self.radio.get_other_bot_collected()) and len(other_bot_collected) == 4 \
                    and self.target.classification == f'{self.color}_box':
                self.collect_state = IDPRobotState.GET_TO_COLLECT_DISTANCE_FROM_BLOCK    # other robot has collected all four targets, this must be ours
            else:  # not able to detect the colour, probably because the position is not accurate
                self.action_queue.insert(1, ("reverse", list(self.get_bot_front(-reverse_distance))))
                self.action_queue.insert(2, "scan")  # rescan to check if there is actually a target there
                # pop the current target off for now, the new scan will give better position
                if self.collect_num_tries > 1:  # this block is probably flipped over
                    self.target.classification = 'flipped'
                    self.collect_num_tries = 0  # reset the counter
                else:
                    self.target_cache.remove_target(self.target)
                    self.collect_num_tries += 1
                self.target = None
                return True

        if self.collect_state == IDPRobotState.GET_TO_COLLECT_DISTANCE_FROM_BLOCK:
            distance_to_move = self.distance_from_bot(self.target.position) - distance_from_block_for_collect
            if abs(distance_to_move) >= self.target_distance_threshold / 4:
                reverse = distance_to_move < 0
                self.drive_to_position(self.get_bot_front(distance_to_move), reverse=reverse)
            else:
                self.collect_state = IDPRobotState.CORRECT_COLOUR

        if self.collect_state == IDPRobotState.CORRECT_COLOUR:
            self.gate.open()
            if self.time - self.stored_time >= gate_time:
                self.collect_state = IDPRobotState.COLLECTING_TARGET

        if self.collect_state == IDPRobotState.COLLECTING_TARGET:
            if self.rotate(rotate_angle, max_rotation_rate=collect_rotation_rate):
                self.stored_time = self.time
                self.collect_state = IDPRobotState.TARGET_COLLECTED

        if self.collect_state == IDPRobotState.TARGET_COLLECTED:
            self.gate.close()
            if self.time - self.stored_time >= gate_time:
                self.target_cache.collected.append(self.target)
                self.target_cache.remove_target(self.target)
                self.target = None
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

                print_if_debug(self.target_cache.targets, debug_flag=DEBUG)

                if self.get_best_target() is None:
                    self.targeting_handler.next_scan_position = self.targeting_handler.get_fallback_scan_position(
                        self.infrared.max_range)
                    self.targeting_handler.relocating = True

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
            print_if_debug(f"\nCompleted action: {self.action_queue[0]}", debug_flag=DEBUG)
            del self.action_queue[0]
            print_if_debug(f"Remaining actions:", debug_flag=DEBUG)

            # Check if action list is now empty
            if len(self.action_queue) == 0:
                print_if_debug("None", debug_flag=DEBUG)
                self.motors.velocities = np.zeros(2)
                return True

            print_if_debug('\n'.join(str(x) for x in self.action_queue), debug_flag=DEBUG)

        return False

    def check_target_valid(self, target: Union[Target, None]) -> bool:
        return target.classification in ['box',
                                         f'{self.color}_box'] and not self.target_cache.check_target_path_blocked(
            target.position,
            self.position,
            self.radio.get_other_bot_position(),
            self.radio.get_other_bot_vertices()
        ) if target else False

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
        """Decide on a new target block for robot, which is the cloest valid target on the way to which
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
