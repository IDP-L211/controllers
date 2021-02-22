# Test script for drive_to_pos

from robot import IDPRobot

# create the Robot instance.
robot = IDPRobot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

action_queue = [
    [1, -1],
    [1, 1],
    3.141,
    [-1, 1],
    [-1, -1],
    [0, 0]
]

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:

    robot.execute_actions(action_queue)

    print(robot.position, robot.speed, robot.bearing)
