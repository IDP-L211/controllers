# Copyright (C) 2021 Weixuan Zhang, Tim Clifford, Jason Brown
#
# SPDX-License-Identifier: MIT
"""This module contains a class representing the robot.
"""

from typing import Union

from controller import Robot
import numpy as np

from devices.sensors import IDPCompass, IDPGPS, IDPDistanceSensor, IDPColorDetector
from devices.motors import IDPMotorController, IDPGate
from devices.radio import IDPRadio

from strategies.motion import MotionCS

from misc.utils import print_if_debug, ensure_list_or_tuple
from misc.geometry import rotate_vector, get_min_distance_rectangles
from misc.mapping import Map
from misc.pid import PID, DataRecorder
from misc.targeting import TargetingHandler, Target, TargetCache

DEBUG = False
tau = np.pi * 2


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

        # Store internal action queue
        self.action_queue = []

        # Store the function associated with each action
        self.action_functions = {
            "move": self.drive_to_position,
            "face": self.face_bearing,
            "rotate": self.rotate,
            "reverse": self.reverse_to_position,
            "collect": self.collect_block,
            "scan": self.scan
        }

        # So we can cleanup if we change our action
        self.last_action_type = None
        self.last_action_value = None

        # State for some composite actions
        self.collect_state = 0
        self.stored_time = 0
        self.num_collected = 0

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
        self.pid_f_velocity = PID("Forward Velocity", self.getTime, 0.1, 0, 0, self.timestep_actual)
        self.pid_distance = PID("Distance", self.getTime, 2, 0, 0, self.timestep_actual)
        self.pid_angle = PID("Angle", self.getTime, 2.5, 0.0, 0.10, self.timestep_actual,
                             derivative_weight_decay_half_life=0.05)

        motor_graph_styles = {"distance": 'k-', "angle": 'r-', "forward_speed": 'k--', "rotation_speed": 'r--',
                              "linear_speed": "k:", "angular_velocity": "r:", "left_motor": 'b-', "right_motor": 'y-'}
        self.motion_history = DataRecorder("time", "distance", "angle", "forward_speed", "rotation_speed", "left_motor",
                                           "right_motor", "linear_speed", "angular_velocity", styles=motor_graph_styles)

    def step(self, timestep):
        """A wrapper for the step call that allows us to keep our last bearing and keep track of time"""
        self.last_bearing = self.bearing if self.time != 0 else None
        self.time += self.timestep_actual
        return super().step(timestep)

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
        self.collect_state = 0

    def drive_to_position(self, target_pos: list, max_forward_speed=None, max_rotation_rate=None,
                          reverse=False) -> bool:
        """Go to a position

        Args:
            target_pos ([float, float]): The East-North co-ords of the target position
            max_forward_speed (float): Maximum speed to travel at m/s
            max_rotation_rate (float): Maximum rate to rotate at, rad/s
            reverse (bool): Whether to reverse there
        Returns:
            bool: If we are at our target
        """
        max_rotation_rate = self.default_max_allowed_speed["r"] if max_rotation_rate is None else max_rotation_rate
        max_rotation_drive = max_rotation_rate / self.max_possible_speed["r"]

        max_forward_speed = self.default_max_allowed_speed["f"] if max_forward_speed is None else max_forward_speed
        max_forward_drive = max_forward_speed / self.max_possible_speed["f"]

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

    def reverse_to_position(self, target_pos: list) -> bool:
        """Go to a position in reverse

        Args:
            target_pos ([float, float]): The East-North co-ords of the target position
        Returns:
            bool: If we are at our target
        """
        return self.drive_to_position(target_pos, reverse=True)

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
        return abs(self.linear_speed) <= self.linear_speed_threshold\
            and abs(self.angular_velocity) <= self.angular_speed_threshold

    def collect_block(self, target: Target):
        """Collect block at position.

        Args:
            target (Target): The target object
        Returns:
            bool: If we are at our target
        """
        distance_from_block_to_stop = 0.15
        max_angle_to_block = 0.1
        rotate_angle = -tau / 4
        gate_time = 0.6
        reverse_distance = 0.3

        # Ifs not elifs means we don't waste timesteps if the state changes

        if self.collect_state == 0:
            if self.distance_from_bot(target.position) - distance_from_block_to_stop >= 0:
                self.drive_to_position(target.position)
            else:
                self.collect_state = 1

        if self.collect_state == 1:
            angle_to_block = self.angle_from_bot_from_position(target.position)
            if abs(angle_to_block) > max_angle_to_block:
                self.rotate(angle_to_block)
            else:
                self.stored_time = self.time
                self.collect_state = 2

        if self.collect_state == 2:
            if self.brake():
                self.collect_state = 3

        if self.collect_state == 3:
            color = self.color_detector.get_color()
            print(f"Block colour: {color}")
            if color in ["red", "green"]:
                target.classification = f"{color}_box"
                if color == self.color:
                    self.collect_state = 4
                else:
                    self.action_queue.insert(1, ("reverse", list(self.get_bot_front(-reverse_distance))))
                    return True
            else:
                self.action_queue.insert(1, ("reverse", list(self.get_bot_front(-reverse_distance))))
                self.action_queue.insert(2, "scan")
                self.target_cache.remove_target(target)
                return True

        if self.collect_state == 4:
            self.gate.open()
            if self.time - self.stored_time >= gate_time:
                self.collect_state = 5

        if self.collect_state == 5:
            if self.rotate(rotate_angle, max_rotation_rate=3.0):
                self.stored_time = self.time
                self.collect_state = 6

        if self.collect_state == 6:
            self.gate.close()
            if self.time - self.stored_time >= gate_time:
                self.target_cache.remove_target(target)
                self.num_collected += 1
                return True

        return False

    def scan(self) -> bool:
        """Rotate 360 degrees to scan for blocks

        Returns:
            bool: Whether the scan is completed
        """
        if not self.targeting_handler.relocating:
            complete = self.rotate(tau, max_rotation_rate=1.0)

            distance = self.infrared.getValue()
            d_min, d_max = self.infrared.getBounds()
            bound = (d_max - d_min) * 2

            if abs(self.get_sensor_distance_to_wall() - distance) > bound * 1.5 \
                    and abs(self.infrared.max_range - distance) > bound:
                self.targeting_handler.positions.append(self.get_bot_front(distance))
                # self.targeting_handler.bounds.append(bound)

            if complete:
                for target_pos in self.targeting_handler.get_targets(self.position):
                    # check if target is the other robot
                    if (other_bot_pos := self.radio.get_other_bot_position()) \
                            and Target.check_near(target_pos, other_bot_pos, 0.3):
                        self.target_cache.add_target(target_pos, classification='robot')
                    else:
                        self.target_cache.add_target(target_pos)

                print_if_debug(self.target_cache.targets)

                if self.get_best_target() is None:
                    self.targeting_handler.next_scan_position = self.targeting_handler.get_fallback_scan_position(
                        self.infrared.max_range)
                    self.targeting_handler.relocating = True

                    # TODO too many scans, probably has collected all the targets, return home

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

        # Check if bot is stuck, note we only reach here if action not completed
        if abs(self.linear_speed) <= self.linear_speed_threshold / 1000\
                and abs(self.angular_velocity) <= self.angular_speed_threshold / 1000\
                and self.collect_state in [1, 0]:
            if self.stuck_steps >= 5:
                print_if_debug(f"BOT STUCK - Attempting unstuck", debug_flag=DEBUG)
                if action_type != "reverse":
                    self.action_queue.insert(0, ("reverse", list(self.coordtransform_bot_polar_to_world(-0.5, 0))))
                else:
                    self.action_queue.insert(0, ("move", list(self.coordtransform_bot_polar_to_world(0.5, 0))))
                self.stuck_steps = 0
            else:
                self.stuck_steps += 1

        return False

    def get_best_target(self) -> Union[Target, None]:
        """Decide on a new target block for robot

        If targeting is not just get closest block there could be more logic here, potentially calls to a script in
            strategies folder that applied more complex algorithms and could even return a list of ordered targets

        For now we just choose the closest block of correct colour or unknown colour

        Returns:
            [float, float]: Targets co-ordinates, East-North, m
        """
        valid_targets_sorted = self.target_cache.get_targets(
            valid_classes=['box', 'red_box', 'green_box'],
            key=lambda t: self.distance_from_bot(t.position)
        )

        for target in valid_targets_sorted:
            if target.classification not in ['box', f'{self.color}_box'] \
                    or self.target_cache.check_target_path_blocked(target, self.position):
                continue  # prevents a block of different colour blocking the path to the target
            return target

        return None
