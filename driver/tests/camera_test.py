# Test script for camera, currently broken as robot.getDevice('camera') returns None

def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())
    camera = robot.getDevice('camera')
    camera.enable(timestep)

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        robot.motors.velocities = [0.5, 0.25]

        print(robot.position, robot.linear_speed, robot.bearing)
        # print(camera.getImageArray())
        robot.map.update()
