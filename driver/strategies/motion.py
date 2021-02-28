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
            np.array(float, float): The speed for the left and right motors respectively. Fraction of max speed.
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
    def angle_based(distance: float, angle: float, r_speed_profile_power=0.5, f_speed_profile_power=3.0) -> np.array:
        """Determine wheels speeds based on the current angle to target, designed to quickly turn to target and then go
        at max forward speed. Could outperform PID control for when a robot doesn't need to stop in an exact spot.

        Args:
            distance (float): Distance from target, m. For this method only matters if it is 0 or not
            angle (float): Angle to target, rad
            r_speed_profile_power (float): Exponent of rotation speed profile, [0, inf]
            f_speed_profile_power (float): How 'tight' to make the velocity profile, [0, inf]

        Returns:
            np.array(float, float): The speed for the left and right motors respectively. Fraction of max speed.
        """
        # For some reason (probably floating point errors), we occasionally get warnings about requested speed exceeding
        # max velocity even though they are equal. We shall subtract a small quantity to avoid this annoyance.
        small_speed = 1e-5

        # Forward speed calculation - aimed to be maximised when facing forward
        forward_speed = (np.cos(angle)**f_speed_profile_power) - small_speed if abs(angle) <= np.pi / 2 else 0

        # Use up the rest of our wheel speed for turning, attenuate to reduce aggressive turning
        rotation_speed = 1 - forward_speed
        rotation_speed = ((abs(rotation_speed))/np.pi)**r_speed_profile_power

        # Zero forward speed if we're not actually needing to move forward
        forward_speed *= np.sign(distance)

        return MotionControlStrategies._combine_and_scale(forward_speed, rotation_speed, angle)
