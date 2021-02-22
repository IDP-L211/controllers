import numpy as np


class MotionControlStrategies:
    """
    All MotionCS methods will return a list of left and right motor velocities
    To be used via robot.motors.velocities(MotionControlStrategies.some_method)
    """

    @staticmethod
    def distance_angle_error(robot) -> list:
        """Set wheel speeds based on angle and distance error

        This method performs well for face_bearing but has issues with point to point travel - slow turn times and
        slowing down a lot when near target

        Returns:
            [float, float]: The speed for the left and right motors respectively to correct both angle and distance
                            error. Given as a fraction of max speed.
        """
        # Control parameters
        k_p_forward = 4
        k_p_rotation = 4

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
    def maximise_a_wheel_speed(robot):
        """Set fastest wheel speed to maximum with the other wheel slowed to facilitate turning

        By using a cos^2 for our forward speed and sin^2 for our rotation speed, the fastest wheel will always be at max
        drive fraction and the other wheel will correct for angle

        This method seems to perform better for point to point travel but is slow to orientate the bot in face_bearing

        Returns:
            [float, float]: The speed for the left and right motors respectively to correct both angle and distance
                            error. Given as a fraction of max speed.
        """
        # For some reason (probably floating point errors), we occasionally get warnings about requested speed exceeding
        # max velocity even though they are equal. We shall subtract a small quantity to avoid this annoyance.
        small_speed = 1e-5

        # How fast our fastest wheel should go
        forward_speed = (np.cos(robot.target_angle) ** 2) - small_speed if abs(robot.target_angle) <= np.pi / 2 else 0

        # How much slower our slower wheel should go for turning purposes
        rotation_speed = (np.sin(robot.target_angle) ** 2) if abs(robot.target_angle) <= np.pi / 2 else 1

        # If we are within distance threshold we no longer need to move forward
        forward_speed = 0 if robot.reached_target else forward_speed

        # If we are within angle threshold we should stop turning
        if robot.target_bearing is None:
            rotation_speed = 0 if robot.reached_target else rotation_speed
        else:
            rotation_speed = 0 if robot.reached_bearing else rotation_speed

        # Combine our speeds based on the sign of our angle
        if robot.target_angle >= 0:
            return [forward_speed + rotation_speed, forward_speed - rotation_speed]
        else:
            return [forward_speed - rotation_speed, forward_speed + rotation_speed]
