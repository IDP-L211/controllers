# Copyright (C) 2021 Jason Brown
#
# SPDX-License-Identifier: MIT
"""Class file for the motion controller"""


import numpy as np


class MotionCS:
    """
    All MotionCS methods will return an array of left and right motor velocities
    To be used via robot.motors.velocities = MotionControlStrategies.some_method(*args)
    """

    # Some characteristics of the robot used for motion calcs
    max_f_speed = 0.5
    max_r_speed = 5.3
    f_drive_speed_ratio = 1 / max_f_speed
    r_drive_speed_ratio = 1 / max_r_speed

    @staticmethod
    def combine_and_scale(forward, rotation, angle=None):
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
    def dual_pid(prime_quantity=None, deriv_quantity=None, prime_pid=None, derivative_pid=None,
                 req_prime_quantity=None, req_deriv_quantity=None) -> float:
        """This method uses 2 PID's to control a quantity based on a threshold. If only one of the required measurements
        is given it just uses a single PID.

        The PID takes a quantity and its derivative (i.e. distance and velocity). When close to closing the error it
        uses the prime quantity but when it's far it uses the derivative quantity and an extra term in the control. Note
        the extra term only applies when controlling via the derivative quantity.

        The extra term is simply the required derivative quantity as a feed-forward PID (open loop + closed loop
        control) is a powerful and responsive control method for situations such as these.

        It switches quantity when the output of the prime quantity controller is equal to the output of the derivative
        quantity controller.

        This function should most likely NOT be called directly, instead call angular_dual_pid or linear_dual_pid which
        act as wrappers for this function that make function args clearer and have useful defaults. These wrappers also
        normalise the values going into the pid so the output is motor drives.

        Notes on the robot:
            Wheel radius: 0.05 m
            Wheel base radius: 0.094 m
            Max turn rate: 5.3 rad/s
            Max forward velocity: 0.50 m/s
            Max rotational acceleration: 2.5 rad/s^2
            Max forward acceleration: 7.9 m/s^2
            Max forward acceleration that produces negligible rotation: 5.0 m/s^2
                Ramp up time: 0.1s
            Max rotational acceleration that produces negligible distance change: 2.4 rad/s^2
                Ramp up time: 2.2s
            Angle rotated stopping at max rotation velocity: 4.8 rad

        Note that calling pid.query simply asks what the output would be if called, whilst pid.step properly calls and
            updates the pid's state

        Args:
            prime_quantity (float): Our current prime quantity. E.g. distance, m or angle, rad
            deriv_quantity (float): Our current derivative quantity. E.g. forward speed, m/s or
                rotational speed, rad/s
            prime_pid (PID): Prime quantity PID controller
            derivative_pid (PID): Derivative quantity PID controller
            req_prime_quantity (float): Required prime quantity
            req_deriv_quantity (float): Required derivative quantity

        Returns:
            float: The drive fraction for the given inputs
        """

        if prime_quantity is None and deriv_quantity is None:
            drive = 0
        else:
            if deriv_quantity is not None:
                derivative_based_drive = derivative_pid.query(req_deriv_quantity - deriv_quantity) + req_deriv_quantity
                if prime_quantity is not None:
                    if abs(prime_pid.query(prime_quantity - req_prime_quantity)) <= abs(derivative_based_drive):
                        drive = prime_pid.step(prime_quantity - req_prime_quantity)
                    else:
                        drive = derivative_pid.step(req_deriv_quantity - deriv_quantity) + req_deriv_quantity
                else:
                    drive = derivative_pid.step(req_deriv_quantity - deriv_quantity) + req_deriv_quantity
            else:
                drive = prime_pid.step(prime_quantity - req_prime_quantity)

        return drive

    @staticmethod
    def angular_dual_pid(angle=None, rotation_rate=None, angle_pid=None, rotational_speed_pid=None, required_angle=0,
                         required_rotation_rate=None):
        """Wrapper for dual_pid to make angular control simpler"""

        # If required_rotation_rate is not specified, use max and then correct for sign of angle
        req_rotation_rate = MotionCS.max_r_speed if required_rotation_rate is None else required_rotation_rate
        req_rotation_rate *= np.sign(angle)

        # Convert from actual robot velocities to drive fraction equivalent
        required_rotational_drive = MotionCS.r_drive_speed_ratio * req_rotation_rate
        rotational_drive_equivalent = None if rotation_rate is None else MotionCS.r_drive_speed_ratio * rotation_rate

        # Return the result from the dual controller
        return MotionCS.dual_pid(prime_quantity=angle, deriv_quantity=rotational_drive_equivalent,
                                 prime_pid=angle_pid, derivative_pid=rotational_speed_pid,
                                 req_prime_quantity=required_angle,
                                 req_deriv_quantity=required_rotational_drive)

    @staticmethod
    def linear_dual_pid(distance=None, forward_speed=None, distance_pid=None, forward_speed_pid=None,
                        required_distance=0, required_forward_speed=None, angle_attenuation=True, angle=None):
        """Wrapper for dual_pid to make linear control simpler"""

        # If required_forward_speed is not specified use max
        req_forward_speed = MotionCS.max_f_speed if required_forward_speed is None else required_forward_speed

        # Attenuate speeds and distance based on angle so robot doesn't zoom off when not facing target
        if angle_attenuation and angle is not None:
            attenuation_factor = (np.cos(angle) ** 3) if abs(angle) <= np.pi / 2 else 0
            distance *= attenuation_factor
            required_distance *= attenuation_factor
            req_forward_speed *= attenuation_factor

        # Convert from actual robot velocities to drive fraction equivalent
        required_forward_drive = MotionCS.f_drive_speed_ratio * req_forward_speed
        forward_speed_drive_eq = None if forward_speed is None else MotionCS.f_drive_speed_ratio * forward_speed

        return MotionCS.dual_pid(prime_quantity=distance, deriv_quantity=forward_speed_drive_eq,
                                 prime_pid=distance_pid, derivative_pid=forward_speed_pid,
                                 req_prime_quantity=required_distance,
                                 req_deriv_quantity=required_forward_drive)
