# Test communication between robots

def main(robot):
    timestep = int(robot.getBasicTimeStep())

    while robot.step(timestep) != -1:
        robot.radio.send_message({'position': robot.position})
        print(f'{robot.color} robot received {robot.radio.get_message()}')
