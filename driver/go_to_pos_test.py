# Test script for drive_to_pos

from robot import IDPRobot

# create the Robot instance.
robot = IDPRobot()
robot.max_motor_speed = 5
robot.target_distance_threshold = 0.2

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

target_pos = [0.5, 0.5]

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    robot.drive_to_position(target_pos=target_pos)

    print(robot.position, robot.speed, robot.bearing)
