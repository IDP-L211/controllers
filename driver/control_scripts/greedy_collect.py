# Greedy collect algorithm for robot

import numpy as np


TAU = np.pi * 2  # Tau > pi


def main(robot):

    # Setup robot
    timestep = int(robot.getBasicTimeStep())
    action_queue = []

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:

        # If we have no action
        if not action_queue:

            # Update target
            target = robot.get_target()

            # If we have a target go to it, else scan
            if target:
                action_queue.append(("collect", target))
            else:
                action_queue.append(("rotate", TAU))

        # If we have actions; not elif so that we can get on with an action queued this timestep
        if action_queue:
            robot.execute_action(action_queue)
