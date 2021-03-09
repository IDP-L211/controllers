# Test script for drive_to_pos

import numpy as np
from modules.utils import fire_and_forget

tau = np.pi * 2


def manual(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())
    robot.step(timestep)

    # Actions for our robot
    action_queue = [
        ("move", [0.5, 0]),
        ("move", [-0.5, 0])
    ]

    robot.action_queue = action_queue

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        robot.execute_next_action()

    fire_and_forget(robot.plot_motion_history)
    fire_and_forget(robot.pid_f_velocity.plot_history)
    fire_and_forget(robot.pid_distance.plot_history)
    fire_and_forget(robot.pid_angle.plot_history)
