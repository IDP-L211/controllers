# Copyright (C) 2021 Weixuan Zhang, Jason Brown
#
# SPDX-License-Identifier: MIT
"""Motors and a class bundling two motors together"""
from controller import Motor, PositionSensor

import numpy as np


class IDPGateMotor(Motor):
    def __init__(self, name):
        super().__init__(name)


class IDPGateSensor(PositionSensor):
    def __init__(self, name, sampling_rate):
        super().__init__(name)
        self.enable(sampling_rate)


class IDPGate:
    uncertainty = 0.02

    def __init__(self, motor_name, sensor_name, sampling_rate):
        self.motor = IDPGateMotor(motor_name)
        self.sensor = IDPGateSensor(sensor_name, sampling_rate)

    def open(self):
        """Opens the robot gate

        Returns:
            bool: Checks if the gate has fully opened
        """
        self.motor.setPosition(1.57)

        return abs(self.sensor.getValue() - self.motor.getTargetPosition()) < IDPGate.uncertainty

    def close(self):
        """Closes the robot gate

        Returns:
            bool: Checks if the gate has fully closed
        """
        self.motor.setPosition(0)

        return abs(self.sensor.getValue() - self.motor.getTargetPosition()) < IDPGate.uncertainty


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
        return np.array(self.left_motor.getVelocity(), self.right_motor.getVelocity())

    @velocities.setter
    def velocities(self, values: np.array):
        """Set the velocities for each motor

        Args:
            values (np.array): Speeds for left and right wheel respectively,
            as fractions of max speed (-1 -> 1), [left, right]
        """
        if len(values) != 2:
            raise Exception("Velocities should be set by a 2 element list")

        values *= self.max_motor_speed
        self.left_motor.setVelocity(values[0])
        self.right_motor.setVelocity(values[1])
