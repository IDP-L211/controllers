import numpy as np
import warnings


class MotionControlStrategies:
    """
    All MotionCS methods will return a list of left and right motor velocities
    To be used via robot.motors.velocities(MotionControlStrategies.some_method)
    """

    @staticmethod
    def distance_angle_error(robot, k_p_forward=4.0, k_p_rotation=4.0) -> list:
        """Set wheel speeds based on angle and distance error

        This method performs well for face_bearing but has issues with point to point travel - slow turn times and
        slowing down a lot when near target

        Args:
            robot: The robot to use for the calculation
            k_p_forward (float): Value of k_p for forward speed
            k_p_rotation (float): Value of k_p for rotation speed

        Returns:
            [float, float]: The speed for the left and right motors respectively. Given as a fraction of max speed.
        """

        forward_speed = robot.target_distance * k_p_forward
        rotation_speed = robot.target_angle * k_p_rotation

        # If we are within distance threshold we no longer need to move forward
        forward_speed = 0 if robot.reached_target else forward_speed

        # If we are within angle threshold we should stop turning
        if robot.target_bearing is None:
            rotation_speed = 0 if robot.reached_target else rotation_speed
        else:
            rotation_speed = 0 if robot.reached_bearing else rotation_speed

        # Attenuate and possibly reverse forward speed based on angle of bot - don't want to speed off away from it
        forward_speed *= np.cos(robot.target_angle)

        # Combine speeds
        speeds = np.array([forward_speed + rotation_speed, forward_speed - rotation_speed])

        # This might be above our motor maximums so we'll use sigmoid to normalise our speeds to this range
        # Sigmoid bounds -inf -> inf to 0 -> 1 so we'll need to do some correcting
        speeds = (1 / (1 + np.exp(-speeds))) * 2 - 1

        # Alternative to using sigmoid
        # speeds = np.tanh(speeds)

        # Another alternative where the minimum motor speed is half its value
        # speeds = (1 / (1 + np.exp(-speeds))) + ((np.sign(speeds) - 1) * 0.5)

        return list(speeds)


    @staticmethod
    def angle_based_control(robot, rotation_speed_profile_power=0.5, forward_speed_profile_power=3.0):
        """Set fastest wheel speed to maximum with the other wheel slowed to facilitate turning

        By using a cos^2 for our forward speed and sin^2 for our rotation speed, the fastest wheel will always be at max
        drive fraction and the other wheel will correct for angle

        This method seems to perform better for point to point travel but is slow to orientate the bot in face_bearing

        Args:
            robot: The robot to use for the calculation
            rotation_speed_profile_power (float): Exponent of rotation speed profile, [0, inf]
            forward_speed_profile_power (float): How 'tight' to make the velocity profile, [0, inf]

        Returns:
            [float, float]: The speed for the left and right motors respectively. Given as a fraction of max speed.
        """
        # For some reason (probably floating point errors), we occasionally get warnings about requested speed exceeding
        # max velocity even though they are equal. We shall subtract a small quantity to avoid this annoyance.
        small_speed = 1e-5

        # Forward speed calculation - aimed to be maximised when facing forward
        forward_speed = (np.cos(robot.target_angle)**forward_speed_profile_power) - small_speed \
            if abs(robot.target_angle) <= np.pi / 2 else 0

        # Use up the rest of our wheel speed for turning, attenuate to reduce aggressive turning
        rotation_speed = 1 - forward_speed
        rotation_speed = ((abs(rotation_speed))/np.pi)**rotation_speed_profile_power

        # If we are within distance threshold we no longer need to move forward
        forward_speed = 0 if robot.reached_target else forward_speed

        # If we are within angle threshold we should stop turning
        if robot.target_bearing is None:
            rotation_speed = 0 if robot.reached_target else rotation_speed
        else:
            rotation_speed = 0 if robot.reached_bearing else rotation_speed

        # Combine our speeds based on the sign of our angle
        if robot.target_angle >= 0:
            left_speed = forward_speed + rotation_speed
            right_speed = forward_speed - rotation_speed
        else:
            left_speed = forward_speed - rotation_speed
            right_speed = forward_speed + rotation_speed

        # Make sure speeds are maxed at 1
        left_speed = min(1, left_speed)
        right_speed = min(1, right_speed)

        return [left_speed, right_speed]

    @staticmethod
    def rotate(robot, rotation_rate: float):
        """Rotate the robot at a fixed rate

        Args:
            robot: The robot to use for the calculation
            rotation_rate (float): Rate of rotation in radians per second, [-inf, inf]

        Returns:
            [float, float]: The speed for the left and right motors respectively. Given as a fraction of max speed."""

        # Calculate wheel drive based on rotation rate
        turn_radius = robot.width / 2
        wheel_drive = (rotation_rate * turn_radius) / (robot.motors.max_motor_speed * robot.wheel_radius)

        if wheel_drive > 1:
            max_rot = rotation_rate/wheel_drive
            warnings.warn(f"Requested rotation rate of {rotation_rate}, exceeds bot's apparent maximum of {max_rot}")

        return [wheel_drive, -wheel_drive]
