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
    def dual_pid(prime_quantity=None, derivative_quantity=None, prime_pid=None, derivative_pid=None,
                 required_prime_quantity=None, required_derivative_quantity=None) -> float:
        """This method uses 2 PID's to control a quantity based on a threshold. If only one of the required measurements
        is given it just uses a single PID.

        The PID takes a quantity and its derivative (i.e. distance and velocity). When close to closing the error it
        uses the prime quantity but when it's far it uses the derivative quantity and an extra term in the control. Note
        the extra term only applies when controlling via the derivative quantity.

        The extra term is simply the required derivative quantity multiplied by a scale factor, if you want to know why
        this is useful, consider the result of a proportional velocity controller who's input is velocity error and
        output is motor velocity. If this still does not clear things after thinking about it for a while then stop
        reading this and pick up a colouring book - it may be more your speed (Pun definitely intended).

        It switches quantity when the proportional output of the prime quantity controller is equal to the total output
        of the derivative quantity controller.

        This function should most likely NOT be called directly, instead call angular_dual_pid or linear_dual_pid which
        act as wrappers for this function that make function args clearer and have useful defaults. These wrappers also
        normalise the values going into the pid so the output is motor drives.

        Note on the robot:
            Max turn rate: 5.3 rad/s
            Max forward velocity: 0.50 m/s

        Args:
            prime_quantity (float): Our current prime quantity. E.g. distance, m or angle, rad
            derivative_quantity (float): Our current derivative quantity. E.g. forward speed, m/s or
                rotational speed, rad/s
            prime_pid (PID): Prime quantity PID controller
            derivative_pid (PID): Derivative quantity PID controller
            required_prime_quantity (float): Required prime quantity
            required_derivative_quantity (float): Required derivative quantity

        Returns:
            float: The drive fraction for the given inputs
        """

        if prime_quantity is None and derivative_quantity is None:
            drive = 0
        else:
            if derivative_quantity is not None:
                drive = derivative_pid.step(required_derivative_quantity - derivative_quantity)\
                        + required_derivative_quantity
                if prime_quantity is not None:
                    distance_proportional_output = prime_pid.k_p * prime_quantity
                    if abs(distance_proportional_output) <= abs(drive):
                        drive = prime_pid.step(prime_quantity - required_prime_quantity)
                        derivative_pid.un_step()
            else:
                drive = prime_pid.step(prime_quantity - required_prime_quantity)

        return drive

    @staticmethod
    def angular_dual_pid(angle=None, rotation_rate=None, angle_pid=None, rotational_speed_pid=None, required_angle=0,
                         required_rotation_rate=5.3):
        """Wrapper for dual_pid to make angular control simpler"""

        # Convert from actual robot velocities to drive fraction equivalent
        required_rotational_drive = 0.189 * required_rotation_rate
        rotational_drive_equivalent = 0.189 * rotation_rate if rotation_rate is not None else None

        return MotionControlStrategies.dual_pid(prime_quantity=angle, derivative_quantity=rotational_drive_equivalent,
                                                prime_pid=angle_pid, derivative_pid=rotational_speed_pid,
                                                required_prime_quantity=required_angle,
                                                required_derivative_quantity=required_rotational_drive)

    @staticmethod
    def linear_dual_pid(distance=None, forward_speed=None, distance_pid=None, forward_speed_pid=None,
                        required_distance=0, required_forward_speed=0.5, angle_attenuation=True, angle=None):
        """Wrapper for dual_pid to make linear control simpler"""

        # Attenuate speeds and distance based on angle so robot doesn't zoom off when not facing target
        if angle_attenuation and angle is not None:
            attenuation_factor = (np.cos(angle) ** 3) if abs(angle) <= np.pi / 2 else 0
            distance *= attenuation_factor
            required_distance *= attenuation_factor
            required_forward_speed *= attenuation_factor

        # Convert from actual robot velocities to drive fraction equivalent
        required_forward_drive = 2.0 * required_forward_speed
        forward_speed_drive_eq = 2.0 * forward_speed if forward_speed is not None else None

        return MotionControlStrategies.dual_pid(prime_quantity=distance, derivative_quantity=forward_speed_drive_eq,
                                                prime_pid=distance_pid, derivative_pid=forward_speed_pid,
                                                required_prime_quantity=required_distance,
                                                required_derivative_quantity=required_forward_drive)

    @staticmethod
    def angle_based(distance: float, angle: float, r_speed_profile_power=0.5, f_speed_profile_power=3.0,
                    combine_speeds=True) -> np.array:
        """Determine wheels speeds based on the current angle to target, designed to quickly turn to target and then go
        at max forward speed. Could outperform PID control for when a robot doesn't need to stop in an exact spot.

        Args:
            distance (float): Distance from target, m. For this method only matters if it is 0 or not
            angle (float): Angle to target, rad
            r_speed_profile_power (float): Exponent of rotation speed profile, [0, inf]
            f_speed_profile_power (float): How 'tight' to make the velocity profile, [0, inf]
            combine_speeds (bool): Whether to combine the speeds and return motor velocities or just give the raw speeds

        Returns:
            np.array(float, float): The speed for the left and right motors respectively. Fraction of max speed.
        """
        # Forward speed calculation - aimed to be maximised when facing forward
        forward_speed = (np.cos(angle)**f_speed_profile_power) if abs(angle) <= np.pi / 2 else 0

        # Use up the rest of our wheel speed for turning, attenuate to reduce aggressive turning
        rotation_speed = 1 - forward_speed
        rotation_speed = ((abs(rotation_speed))/np.pi)**r_speed_profile_power

        # Zero forward speed if we're not actually needing to move forward
        forward_speed *= np.sign(distance)

        if not combine_speeds:
            return forward_speed, rotation_speed
        else:
            return MotionControlStrategies.combine_and_scale(forward_speed, rotation_speed, angle)
