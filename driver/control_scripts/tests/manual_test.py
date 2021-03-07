
import numpy as np


def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())

    # In-case of ramp-up tests
    ramp_up_time = 2.5
    final_speed = 1
    robot.step(timestep)
    robot.time = 0

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:

        if robot.time <= 5:
            speed = final_speed * min(1.0, robot.time / ramp_up_time)
            motor_velocities = np.array([speed, -speed])
        else:
            motor_velocities = np.zeros(2)

        robot.motors.velocities = motor_velocities
        robot.update_motion_history(time=robot.time, linear_speed=robot.linear_speed,
                                    angular_velocity=robot.angular_velocity)

    robot.plot_motion_history()
