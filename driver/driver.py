# Copyright (C) 2021 Weixuan Zhang, Tim Clifford, Jason Brown
#
# SPDX-License-Identifier: MIT
"""Main driver code
"""
from robot import IDPRobot

# Change which script you want to run here
from control_scripts.motion_test import main as motion_test
from control_scripts.camera_test import main as camera_test
from control_scripts.object_processing_test import main as object_processing_test
from control_scripts.sensor_bounds_test import main as sensor_bounds_test


if __name__ == '__main__':
    robot = IDPRobot()
    # motion_test(robot)
    # camera_test(robot)
    # object_processing_test(robot)
    sensor_bounds_test(robot)
