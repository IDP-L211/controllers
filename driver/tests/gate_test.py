# Test script for opening and closing gate

def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:

        if not(gate_open := robot.gate.open()):
            print(gate_open)
