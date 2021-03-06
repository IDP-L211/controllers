# Copyright (C) 2021 Weixuan Zhang, Eleanor Clifford, Jason Brown
#
# SPDX-License-Identifier: MIT
"""Main driver code
"""
from robot import IDPRobot
from supervisor import IDPSupervisor

# Change which script you want to run here
from control_scripts.tests.motion_test import main as motion_test
from control_scripts.tests.com_test import main as com_test
from control_scripts.tests.camera_test import main as camera_test
from control_scripts.tests.object_processing_test import main as object_processing_test
from control_scripts.tests.block_collect_test import main as block_collect_test
from control_scripts.tests.sensor_dist_calc_test import main as sensor_dist_calc_test
from control_scripts.tests.sensor_bounds_ir_test import main as sensor_bounds_ir_test
from control_scripts.tests.sensor_bounds_ultrasonic_test import main as sensor_bounds_ultrasonic_test
from control_scripts.greedy_collect import main as greedy_collect
from pid_tuning import manual as pid_tuning
from control_scripts.tests.manual_test import main as manual_test


if __name__ == '__main__':
    # robot = IDPSupervisor()
    robot = IDPRobot()
    # com_test(robot)
    # motion_test(robot)
    # camera_test(robot)
    # object_processing_test(robot)
    # block_collect_test(robot)
    # sensor_dist_calc_test(robot)
    # sensor_bounds_ultrasonic_test(robot)
    # sensor_bounds_ir_test(robot)
    # greedy_collect(robot)
    # pid_tuning(robot)
    # manual_test(robot)
    greedy_collect(robot)
