# Copyright (C) 2021 Weixuan Zhang, Eleanor Clifford, Jason Brown, Ghifari Pradana
#
# SPDX-License-Identifier: MIT
"""This is the driver script of the controller
"""
from robot import IDPRobot

robot = IDPRobot()
complete = False

# Main loop, perform simulation steps until Webots is stopping the controller
while robot.step(robot.timestep) != -1:

    # Check if we are finished
    if robot.target_cache.num_collected >= 4 or robot.targeting_handler.num_scans >= 10:
        if robot.distance_from_bot(robot.home) <= 0.1:
            robot.do("hold")
            if not complete:
                complete = True
                print(f"{robot.color} complete")
        else:
            robot.do("move", robot.home)

    # If we have no action
    if robot.execute_next_action():
        # Update target
        target = robot.get_best_target()

        # If we have a target go to it, else scan
        if target:
            robot.do("collect", target)
        else:
            robot.do("scan")
