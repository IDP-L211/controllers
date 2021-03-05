# Copyright (C) 2021 Weixuan Zhang, Jason Brown
#
# SPDX-License-Identifier: MIT
"""Motors and a class bundling two motors together"""
from controller import Motor

import numpy as np


class IDPMotor(Motor):
    def __init__(self, name):
        super().__init__(name)
        self.setPosition(float('inf'))
        self.setVelocity(0.0)


class IDPMotorController:

    max_f_speed = 0.5
    max_r_speed = 5.3
    max_f_acc = 5.0
    max_r_acc = 2.5

    def __init__(self, left_motor_name: str, right_motor_name: str, robot):
        self.robot = robot
        self.left_motor = IDPMotor(left_motor_name)
        self.right_motor = IDPMotor(right_motor_name)
        self.max_motor_speed = min(self.left_motor.getMaxVelocity(), self.right_motor.getMaxVelocity())
        self.last_r_speed = 0
        self.last_f_speed = 0

    @property
    def velocities(self):
        return np.array([self.right_motor.getVelocity(), self.left_motor.getVelocity()]) / self.max_motor_speed

    @velocities.setter
    def velocities(self, drive_fractions: np.array):
        """Set the velocities for each motor

        Args:
            drive_fractions (np.array): Speeds for left and right wheel respectively,
            as fractions of max speed (-1 -> 1), [left, right]
        """
        if len(drive_fractions) != 2:
            raise Exception("Velocities should be set by a 2 element array")

        # Limit motor velocities to avoid slip
        # Derive forward and rotational speeds
        f_speed = 0.5 * sum(drive_fractions) * self.max_f_speed
        r_speed = 0.5 * (drive_fractions[0] - drive_fractions[1]) * self.max_r_speed

        # Limit forward speed
        max_f_speed = self.last_f_speed + (self.max_f_acc * self.robot.timestep_actual)
        min_f_speed = self.last_f_speed - (self.max_f_acc * self.robot.timestep_actual)
        f_speed = max(min(f_speed, max_f_speed), min_f_speed)

        # Limit rotational speed
        max_r_speed = self.last_r_speed + (self.max_r_acc * self.robot.timestep_actual)
        min_r_speed = self.last_r_speed - (self.max_r_acc * self.robot.timestep_actual)
        r_speed = max(min(r_speed, max_r_speed), min_r_speed)

        self.last_f_speed = f_speed
        self.last_r_speed = r_speed

        # Recombine to get drive fractions
        f_drive = f_speed / self.max_f_speed
        r_drive = r_speed / self.max_r_speed
        drive_fractions = np.array([f_drive + r_drive, f_drive - r_drive])

        # Scale values
        values = self.max_motor_speed * drive_fractions

        # TODO - Put these back when model changes
        self.left_motor.setVelocity(values[1])
        self.right_motor.setVelocity(values[0])
