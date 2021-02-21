# Test script for drive_to_pos

from robot import IDPRobot

# create the Robot instance.
robot = IDPRobot()
robot.max_motor_speed = 5
robot.target_distance_threshold = 0.2

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

target_pos = [0.5, 0.5]
bearing = 3.141
at_target = False

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    bearing_override = bearing if at_target else None
    at_target = robot.drive_to_position(target_pos=target_pos, target_bearing_override=bearing_override)
    print(at_target)
    print(robot.position, robot.speed, robot.bearing)
