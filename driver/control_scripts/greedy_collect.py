# Greedy collect algorithm for robot

import numpy as np


TAU = np.pi * 2  # Tau > pi


def main(robot):

    # Setup robot
    timestep = int(robot.getBasicTimeStep())
    nav_map = robot.get_map(robot.infrared)

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:

        # If we have no action
        if robot.execute_next_action():
            # Update target
            target = robot.get_best_target()

            # If we have a target go to it, else scan
            if target:
                robot.do("collect", target)
            else:
                robot.do("scan")

        for t in robot.target_cache.targets:
            nav_map.plot_coordinate(t.position)
        nav_map.update()

