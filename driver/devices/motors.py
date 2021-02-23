# Copyright (C) 2021 Weixuan Zhang, Jason Brown
#
# SPDX-License-Identifier: MIT
"""Motors and a class bundling two motors together"""
from controller import Motor


class IDPMotor(Motor):
    def __init__(self, name):
        super().__init__(name)
        self.setPosition(float('inf'))
        self.setVelocity(0.0)


class IDPMotorController:
    def __init__(self, left_motor_name: str, right_motor_name: str):
        self.left_motor = IDPMotor(left_motor_name)
        self.right_motor = IDPMotor(right_motor_name)
        self.max_motor_speed = min(self.left_motor.getMaxVelocity(), self.right_motor.getMaxVelocity())

    @property
    def velocities(self):
        return [self.left_motor.getVelocity(), self.right_motor.getVelocity()]

    @velocities.setter
    def velocities(self, value: list):
        """Set the velocities for each motor

        Args:
            value (list): Speeds for left and right wheel respectively, as fractions of max speed (-1 -> 1), [left, right]
        """
        if len(value) != 2:
            raise Exception("Velocities should be set by a 2 element list")

        left_speed = value[0] * self.max_motor_speed
        right_speed = value[1] * self.max_motor_speed

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
