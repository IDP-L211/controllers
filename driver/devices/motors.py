# Copyright (C) 2021 Weixuan Zhang, Jason Brown
#
# SPDX-License-Identifier: MIT
"""Motors and a class bundling two motors together"""
from controller import Motor

import numpy as np


class IDPGate(Motor):
    uncertainty = 0.02

    def __init__(self, name):
        super().__init__(name)

    def open(self):
        """Opens the robot gate"""
        self.setPosition(np.pi / 2)

    def close(self):
        """Closes the robot gate"""
        self.setPosition(0)


class IDPMotor(Motor):
    def __init__(self, name):
        super().__init__(name)
        self.setPosition(float('inf'))
        self.setVelocity(0.0)


class IDPMotorController:

    def __init__(self, left_motor_name: str, right_motor_name: str, robot):
        self.robot = robot
        self.left_motor = IDPMotor(left_motor_name)
        self.right_motor = IDPMotor(right_motor_name)
        self.max_motor_speed = min(self.left_motor.getMaxVelocity(), self.right_motor.getMaxVelocity())
        self.last_speed = {"f": 0, "r": 0}  # TODO - Occasionally sync these with robot's velocities

    @property
    def velocities(self):
        return np.array([self.left_motor.getVelocity(), self.right_motor.getVelocity()]) / self.max_motor_speed

    @velocities.setter
    def velocities(self, drive_fractions: np.array):
        """Set the velocities for each motor

        Args:
            drive_fractions (np.array): Speeds for left and right wheel respectively,
            as fractions of max speed (-1 -> 1), [left, right]
        """
        if len(drive_fractions) != 2:
            raise Exception("Velocities should be set by a 2 element array")

        # Reconstitute forward and rotational velocities
        f_speed = 0.5 * sum(drive_fractions)
        r_speed = 0.5 * (drive_fractions[0] - drive_fractions[1])

        # Process them to limit motor velocity changes
        def limit_velocity_change(drive, speed_type):
            speed = drive * self.robot.max_possible_speed[speed_type]
            max_speed = self.last_speed[speed_type] + (self.robot.max_acc[speed_type] * self.robot.timestep_actual)
            min_speed = self.last_speed[speed_type] - (self.robot.max_acc[speed_type] * self.robot.timestep_actual)
            speed = max(min(speed, max_speed), min_speed)
            self.last_speed[speed_type] = speed
            return speed / self.robot.max_possible_speed[speed_type]
        f_drive = limit_velocity_change(f_speed, "f")
        r_drive = limit_velocity_change(r_speed, "r")

        # Reassemble drive and convert to motor values
        values = np.array([f_drive + r_drive, f_drive - r_drive]) * self.max_motor_speed

        # TODO - Put these back when model changes
        self.left_motor.setVelocity(values[0])
        self.right_motor.setVelocity(values[1])
