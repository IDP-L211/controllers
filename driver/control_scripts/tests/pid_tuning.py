# Test script for drive_to_pos

import numpy as np

tau = np.pi * 2


def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())
    robot.step(timestep)

    # Actions for our robot
    action_queue = [
        ("move", [0.5, 0])
    ]

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        robot.execute_action(action_queue)

    robot.pid_f_velocity.evaluate("error", "error_change", "cumulative_error")
    robot.pid_distance.evaluate("error", "error_change", "cumulative_error",
                                settle_threshold=robot.target_distance_threshold)
    robot.pid_angle.evaluate("error", "error_change", "cumulative_error",
                             settle_threshold=robot.target_bearing_threshold)
