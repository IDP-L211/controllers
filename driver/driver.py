"""Main controller

The robot should have GPS (named 'gps) and Compass (named 'compass') added as children
"""

from robot import IDPRobot

# create the Robot instance.
robot = IDPRobot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# camera = robot.getDevice('camera')
# camera.enable(timestep)

nav_map = robot.get_map(2.4)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    robot.left_motor.setVelocity(2.0)
    robot.right_motor.setVelocity(1.0)

    print(f"[{robot.position[0]:4.2f} {robot.position[1]:4.2f} ] "  + \
          f"{robot.speed:4.2f} {robot.bearing:4.2f} " + \
          f"{robot.ultrasonic.getValue():4.2f}")

    # print(camera.getImageArray())

    nav_map.update()
