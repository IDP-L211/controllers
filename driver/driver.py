# from controller import Robot, Motor, DistanceSensor
from robot import IDPRobot

# create the Robot instance.
robot = IDPRobot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    robot.left_motor.setVelocity(10.0)

    print(robot.position, robot.speed, robot.bearing)
