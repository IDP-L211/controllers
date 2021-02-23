# Copyright (C) 2021 Weixuan Zhang
# Copyright (C) 2021 Tim Clifford
#
# SPDX-License-Identifier: MIT
from robot import IDPRobot

# Change which script you want to run here
from control_scripts.motion_test import main as motion_test
from control_scripts.camera_test import main as camera_test


if __name__ == '__main__':
    robot = IDPRobot()
    motion_test(robot)
