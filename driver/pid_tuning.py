# Test script for drive_to_pos

import numpy as np

tau = np.pi * 2


def manual(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())
    robot.step(timestep)

    # Actions for our robot
    action_queue = [
        ("rotate", tau)
    ]

    robot.action_queue = action_queue

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        robot.execute_next_action()

    robot.plot_all_graphs()
