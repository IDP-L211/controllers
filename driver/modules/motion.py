# Copyright (C) 2021 Jason Brown
#
# SPDX-License-Identifier: MIT
"""Class file for the motion controller"""


import numpy as np

tau = np.pi * 2


class MotionCS:
    """
    All MotionCS methods will return an array of left and right motor velocities
    To be used via robot.motors.velocities = MotionControlStrategies.some_method(*args)
    """

    # Overwritten in robot.py
    max_f_speed = None

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
        uses the prime quantity but when it's far it uses the derivative quantity

        With derivative control it acts as a feed-forward PID (open loop + closed loop) as this is a powerful and
        responsive control method for situations such as these.

        It switches quantity when the output of the prime quantity controller is equal to the output of the derivative
        quantity controller, this is found using pid.query which gets output without affecting pid state.

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
                    if np.sign(derivative_based_drive) * prime_pid.query(prime_quantity - req_prime_quantity) <= abs(derivative_based_drive):
                        drive = prime_pid.step(prime_quantity - req_prime_quantity)
                    else:
                        drive = derivative_pid.step(req_deriv_quantity - deriv_quantity) + req_deriv_quantity
                else:
                    drive = derivative_pid.step(req_deriv_quantity - deriv_quantity) + req_deriv_quantity
            else:
                drive = prime_pid.step(prime_quantity - req_prime_quantity)

        return drive

    @staticmethod
    def linear_dual_pid(distance=None, forward_speed=None, distance_pid=None, forward_speed_pid=None,
                        required_distance=0, required_forward_speed=None, angle_attenuation=True, angle=None):
        """Wrapper for dual_pid to make linear control simpler"""

        # If required_forward_speed is not specified use max
        req_forward_speed = MotionCS.max_f_speed if required_forward_speed is None else required_forward_speed

        # Attenuate speeds and distance based on angle so robot doesn't zoom off when not facing target
        if angle_attenuation and angle is not None:
            attenuation_factor = (np.cos(angle) ** 5) if abs(angle) <= tau / 4 else 0
            distance *= attenuation_factor
            required_distance *= attenuation_factor
            req_forward_speed *= attenuation_factor

        # Convert from actual robot velocities to drive fraction equivalent
        required_forward_drive = req_forward_speed / MotionCS.max_f_speed
        forward_speed_drive_eq = None if forward_speed is None else forward_speed / MotionCS.max_f_speed

        return MotionCS.dual_pid(prime_quantity=distance, deriv_quantity=forward_speed_drive_eq,
                                 prime_pid=distance_pid, derivative_pid=forward_speed_pid,
                                 req_prime_quantity=required_distance, req_deriv_quantity=required_forward_drive)
