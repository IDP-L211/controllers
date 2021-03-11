# Copyright (C) 2021 Weixuan Zhang, Tim Clifford, Jason Brown, Ghifari Pradana
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
        # If we have a target go to it, else scan
        if robot.get_best_target():
            robot.do("collect")
        else:
            other_bot_collected = robot.radio.get_other_bot_collected()
            if other_bot_collected is not None and len(other_bot_collected) == 4:
                robot.target_cache.update_flipped(robot.color)
            robot.do("scan")
