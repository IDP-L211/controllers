# Greedy collect algorithm for robot

import numpy as np


TAU = np.pi * 2  # Tau > pi


def main(robot):

    # Setup robot
    timestep = int(robot.getBasicTimeStep())
    action_queue = []

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:

        # Code

        # If we have no action
        if not action_queue:
            # Code
            pass

        # Code

        # If we have actions; not elif so that we can get on with an action queued this timestep
        if action_queue:
            # Code
            robot.execute_action(action_queue)
            # Code

        # Code
