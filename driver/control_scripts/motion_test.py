# Test script for drive_to_pos

from driver.robot.core import IDPRobot

# create the Robot instance.
robot = IDPRobot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

action_queue = [
    ("move", [1, -1]),
    ("move", [1, 1]),
    ("face", 3.141),
    ("move", [-1, 1]),
    ("move", [-1, -1]),
    ("move", [0, 0]),
    ("face", 0.0)
]

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:

    robot.execute_actions(action_queue)

    print(robot.position, robot.speed, robot.bearing)
