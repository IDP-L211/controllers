# Test script for camera, currently broken as robot.getDevice('camera') returns None

def main(robot):

    # Setup
    timestep = int(robot.getBasicTimeStep())
    camera = robot.getDevice('camera')
    camera.enable(timestep)
    nav_map = robot.get_map(2.4)

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:

        robot.left_motor.setVelocity(2.0)
        robot.right_motor.setVelocity(1.0)

        print(robot.position, robot.speed, robot.bearing)
        # print(camera.getImageArray())
        nav_map.update()
