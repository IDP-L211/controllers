# Test script for drive_to_pos

def main(robot):
    timestep = int(robot.getBasicTimeStep())
    while robot.step(timestep) != -1:
        print(f"RED: {robot.red_light_sensor.getValue():.4f}",
              f"GREEN: {robot.green_light_sensor.getValue():.4f}")
