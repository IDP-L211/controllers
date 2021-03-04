
import numpy as np


def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())

    # In-case of ramp-up tests
    ramp_up_time = 1
    final_speed = 1

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        speed = final_speed * min(1, robot.time / ramp_up_time)
        robot.motors.velocities = np.array([1, 1])
        print(robot.linear_speed, robot.angular_velocity)
