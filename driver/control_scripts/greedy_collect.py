# Greedy collect algorithm for robot

import numpy as np


TAU = np.pi * 2  # Tau > pi


def main(robot):

    # Setup robot
    timestep = int(robot.getBasicTimeStep())

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:

        # If we have no action
        if robot.idle:

            # Update target
            target = robot.get_target()

            # If we have a target go to it, else scan
            if target:
                robot.do("collect", target)
            else:
                robot.do("rotate", TAU)

        # If we have actions; not elif so that we can get on with an action queued this timestep
        if not robot.idle:
            robot.execute_next_action()
