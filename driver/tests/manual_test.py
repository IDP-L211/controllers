
import numpy as np


def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())

    robot.step(timestep)
    robot.time = 0

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:

        robot.motors.velocities = np.array([1, 1])
        robot.update_motion_history(time=robot.time, linear_speed=robot.linear_speed,
                                    angular_velocity=robot.angular_velocity)

    robot.plot_motion_history()
