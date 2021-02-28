# Copyright (C) 2021 Jason Brown
#
# SPDX-License-Identifier: MIT
"""Class file for the motion controller"""


import numpy as np


class MotionControlStrategies:
    """
    All MotionCS methods will return an array of left and right motor velocities
    To be used via robot.motors.velocities = MotionControlStrategies.some_method(*args)
    """

    @staticmethod
    def _combine_and_limit(forward, rotation, angle=None):
        # Reverse rotation if angle is negative for symmetric strategies
        if angle is not None:
            rotation *= np.sign(angle)

        # Make sure speeds are maxed at 1
        left_speed = max(min(1, forward + rotation), -1)
        right_speed = max(min(1, forward - rotation), -1)

        return np.array([left_speed, right_speed])

    @staticmethod
    def _combine_and_scale(forward, rotation, angle=None):
        # Reverse rotation if angle is negative for symmetric strategies
        if angle is not None:
            rotation *= np.sign(angle)

        # Combine velocities
        left_speed = forward + rotation
        right_speed = forward - rotation

        # Scale so max is +-1
        if abs(left_speed) >= 1:
            left_speed = left_speed / abs(left_speed)
            right_speed = right_speed / abs(left_speed)

        if abs(right_speed) >= 1:
            right_speed = right_speed / abs(right_speed)
            left_speed = left_speed / abs(right_speed)

        return np.array([left_speed, right_speed])

    @staticmethod
    def combined_pid(current_f_velocity=None, current_distance=None, current_r_velocity=None, current_angle=None,
                     pid_f_velocity=None, pid_distance=None, pid_r_velocity=None, pid_angle=None,
                     required_f_velocity=0.4, required_distance=0.0, required_r_velocity=8.0, required_angle=0.0,
                     switch_distance=0.2, switch_angle=np.pi):
        """This method uses PID control for four different quantities based on thresholds. If some quantities aren't
        given then it will try and use the other quantity for that velocity type. If that quantity is also not given it
        assumes that no velocity of that type is required. As a result this strategy can be used for any situation.

        For virtually all use cases the default value of required_distance and required_angle will suffice.

        The required velocities are defaulted to maximum possible for the robot

        Args:
            current_f_velocity (float): Our current forward velocity, m/s
            current_distance (float): Our current distance from the target, m
            current_r_velocity (float): Our current rotational velocity, rad/s
            current_angle (float): Our current angle from the target, rad
            pid_f_velocity (PID): Forward velocity PID controller
            pid_distance (PID): Distance PID controller
            pid_r_velocity (PID): Rotational velocity PID controller
            pid_angle (PID): Angle PID controller
            required_f_velocity (float): Required forward velocity, m/s
            required_distance (float): Required distance from the target, m
            required_r_velocity (float): Required rotational velocity, rad/s
            required_angle (float): Required angle from target, rad
            switch_distance (float): Within this distance from target use distance error, else forward velocity, m
            switch_angle (float): Within this angle from target use angle error, else rotational velocity, rad

        Returns:
            np.array(float, float): The speed for the left and right motors respectively. Given as a fraction of max speed.
        """

        if current_distance is None and current_f_velocity is None:
            forward_speed = 0
        else:
            if (current_distance is None or current_distance > switch_distance) and current_f_velocity is not None:
                forward_speed = pid_f_velocity.step(required_f_velocity - current_f_velocity)
            else:
                forward_speed = pid_distance.step(current_distance - required_distance)

        if current_angle is None and current_r_velocity is None:
            rotation_speed = 0
        else:
            if (current_angle is None or current_angle > switch_angle) and current_r_velocity is not None:
                rotation_speed = pid_r_velocity.step(required_r_velocity - current_r_velocity)
            else:
                rotation_speed = pid_angle.step(current_angle - required_angle)

        return MotionControlStrategies._combine_and_scale(forward_speed, rotation_speed)

    @staticmethod
    def distance_angle_error(distance: float, angle: float, k_p_forward=4.0, k_p_rotation=4.0) -> np.array:
        """Set wheel speeds based on angle and distance error

        This method performs well for face_bearing but has issues with point to point travel - slow turn times and
        slowing down a lot when near target

        Args:
            distance (float): Distance to target, m
            angle (float): Angle to target, rad
            k_p_forward (float): Value of k_p for forward speed
            k_p_rotation (float): Value of k_p for rotation speed

        Returns:
            np.array(float, float): The speed for the left and right motors respectively. Given as a fraction of max speed.
        """

        forward_speed = distance * k_p_forward
        rotation_speed = angle * k_p_rotation

        # Attenuate and possibly reverse forward speed based on angle of bot - don't want to speed off away from it
        forward_speed *= np.cos(angle)

        # Combine speeds
        speeds = np.array([forward_speed + rotation_speed, forward_speed - rotation_speed])

        # This might be above our motor maximums so we'll use sigmoid to normalise our speeds to this range
        # Sigmoid bounds -inf -> inf to 0 -> 1 so we'll need to do some correcting
        speeds = (1 / (1 + np.exp(-speeds))) * 2 - 1

        # Alternative to using sigmoid
        # speeds = np.tanh(speeds)

        # Another alternative where the minimum motor speed is half its value
        # speeds = (1 / (1 + np.exp(-speeds))) + ((np.sign(speeds) - 1) * 0.5)

        return speeds

    @staticmethod
    def angle_based_control(distance: float, angle: float, rotation_speed_profile_power=0.5,
                            forward_speed_profile_power=3.0) -> np.array:
        """Set fastest wheel speed to maximum with the other wheel slowed to facilitate turning

        By using a cos^2 for our forward speed and sin^2 for our rotation speed, the fastest wheel will always be at max
        drive fraction and the other wheel will correct for angle

        This method seems to perform better for point to point travel but is slow to orientate the bot in face_bearing

        Args:
            distance (float): Distance from target, m. For this method only matters if it is 0 or not
            angle (float): Angle to target, rad
            rotation_speed_profile_power (float): Exponent of rotation speed profile, [0, inf]
            forward_speed_profile_power (float): How 'tight' to make the velocity profile, [0, inf]

        Returns:
            np.array(float, float): The speed for the left and right motors respectively. Given as a fraction of max speed.
        """
        # For some reason (probably floating point errors), we occasionally get warnings about requested speed exceeding
        # max velocity even though they are equal. We shall subtract a small quantity to avoid this annoyance.
        small_speed = 1e-5

        # Forward speed calculation - aimed to be maximised when facing forward
        forward_speed = (np.cos(angle)**forward_speed_profile_power) - small_speed if abs(angle) <= np.pi / 2 else 0

        # Use up the rest of our wheel speed for turning, attenuate to reduce aggressive turning
        rotation_speed = 1 - forward_speed
        rotation_speed = ((abs(rotation_speed))/np.pi)**rotation_speed_profile_power

        # Zero forward speed if we're not actually needing to move forward
        forward_speed *= np.sign(distance)

        return MotionControlStrategies._combine_and_limit(forward_speed, rotation_speed, angle)

    @staticmethod
    def short_linear_region(distance, angle, forward_drive=1, angle_drive=1,
                            forward_lin_region_width=0.2, angular_lin_region_width=np.pi/10) -> np.array:
        """Calculates wheel speeds based on trying to achieve a fixed drive but with a short linear region when close

        Args:
            distance (float): Distance from target, m. For this method only matters if it is 0 or not
            angle (float): Angle to target, rad
            forward_drive (float): Ideal forward drive when far from target [0, 1]
            angle_drive (float): Ideal angle drive when far from target [0, 1]
            forward_lin_region_width (float): Width of the linear region for forward drive
            angular_lin_region_width (float): Width of the linear region for angular drive

        Returns:
            [float, float]: The speed for the left and right motors respectively. Given as a fraction of max speed."""

        forward_speed = forward_drive if abs(distance) > forward_lin_region_width else\
            (distance / forward_lin_region_width) * forward_drive
        rotation_speed = angle_drive if abs(angle) > angular_lin_region_width else\
            (angle / angular_lin_region_width) * angle_drive

        return MotionControlStrategies._combine_and_limit(forward_speed, rotation_speed, angle)
