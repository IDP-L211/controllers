# Test script for drive_to_pos

import numpy as np

tau = np.pi * 2


def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())

    # Actions for our robot
    action_queue = [
        ("rotate", tau)
    ]

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        robot.execute_action(action_queue)

    robot.pid_angle.plot("error")
