# Test script for drive_to_pos

import numpy as np
from misc.utils import fire_and_forget

tau = np.pi * 2


def trial(supervisor):
    # Setup robot
    robot_node = supervisor.getFromDef("MY_ROBOT")
    timestep = int(supervisor.getBasicTimeStep())

    # Reset bot to center bottom, facing east
    robot_node.getField("translation").setSFVec3f([-0.75, 0.035, 0])
    robot_node.getField("rotation").setSFRotation([0, 1, 0, 0])
    robot_node.resetPhysics()

    # The actions for our bot to perform
    test_actions = [
        ("rotate", tau/2),
        ("face", tau/2),
        ("move", [0, 0.75]),
        ("move", [0.5, 0.75])
    ]

    # Load actions and store timings
    supervisor.action_queue = test_actions
    start_time = supervisor.getTime()
    max_allowed_time = 10

    # Execute until finished or out of time
    while supervisor.getTime() - start_time < max_allowed_time:
        if supervisor.step(timestep) == -1 or supervisor.execute_next_action():
            break

    # Return the time taken
    return supervisor.getTime() - start_time


def genetic_optimise(supervisor):
    trial(supervisor)


def manual(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())
    robot.step(timestep)

    # Actions for our robot
    action_queue = [
        ("face", 0),
        ("move", [0.5, 0])
    ]

    robot.action_queue = action_queue

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        robot.execute_next_action()

    fire_and_forget(robot.evaluate)
    fire_and_forget(robot.pid_f_velocity.evaluate)
    fire_and_forget(robot.pid_distance.evaluate)
    fire_and_forget(robot.pid_angle.evaluate)
