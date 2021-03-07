# Test script for opening and closing gate

def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())
    gate_close = False

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:

        if gate_close:
            robot.open_gate()
        else:
            robot.close_gate()

        '''else:
            robot.close_gate

        gate_open != gate_open '''
