# Copyright (C) 2021 Weixuan Zhang, Eleanor Clifford, Jason Brown
#
# SPDX-License-Identifier: MIT
"""Main driver code
"""
from robot import IDPRobot

# Change which script you want to run here
from tests.motion_test import main as motion_test
from tests.com_test import main as com_test
from tests.camera_test import main as camera_test
from tests.gate_test import main as gate_test
from tests.object_processing_test import main as object_processing_test
from tests.block_collect_test import main as block_collect_test
from tests.sensor_dist_calc_test import main as sensor_dist_calc_test
from tests.sensor_bounds_ir_test import main as sensor_bounds_ir_test
from tests.sensor_bounds_ultrasonic_test import main as sensor_bounds_ultrasonic_test
from pid_tuning import manual as pid_tuning
from tests.manual_test import main as manual_test
from tests.color_test import main as color_test
from tests.collision_avoidance import main as collision_avoidance_test

if __name__ == '__main__':
    robot = IDPRobot()
    # com_test(robot)
    motion_test(robot)
    # camera_test(robot)
    # gate_test(robot)
    # object_processing_test(robot)
    # block_collect_test(robot)
    # sensor_dist_calc_test(robot)
    # sensor_bounds_ultrasonic_test(robot)
    # sensor_bounds_ir_test(robot)
    # greedy_collect(robot)
    # pid_tuning(robot)
    # manual_test(robot)
    # color_test(robot)
    collision_avoidance_test(robot)
