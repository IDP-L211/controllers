import numpy as np


class MotionControlStrategies:
    """
    All MotionCS methods will return am array of left and right motor velocities
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
    def f_velocity_angle_pid(distance: float, angle: float, current_f_velocity, pid_f_velocity, pid_angle):

        velocity_error = np.tanh(distance * 10) - current_f_velocity

        forward_speed = pid_f_velocity.step(velocity_error)
        rotation_speed = pid_angle.step(angle)

        return MotionControlStrategies._combine_and_scale(forward_speed, rotation_speed)

    @staticmethod
    def f_r_velocity_pid(current_f_velocity, current_r_velocity, required_f_velocity, required_r_velocity,
                         pid_f_velocity, pid_r_velocity):

        f_velocity_error = required_f_velocity - current_f_velocity
        r_velocity_error = required_r_velocity - current_r_velocity

        forward_speed = pid_f_velocity.step(f_velocity_error)
        rotation_speed = pid_r_velocity.step(r_velocity_error)

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
