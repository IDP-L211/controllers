# Test script for drive_to_pos

from robot import IDPRobot

# create the Robot instance.
robot = IDPRobot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

target_pos = [0.5, 0.5]
bearing = 3.141
at_target = False

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:

    if not at_target:
        at_target = robot.drive_to_position(target_pos=target_pos)
    else:
        robot.face_bearing(bearing)

    print(robot.position, robot.speed, robot.bearing)
