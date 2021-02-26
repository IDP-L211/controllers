# Test script for collecting a block

def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())
    nav_map = robot.get_map(robot.ir_long, 2.4)

    # Actions for our robot
    robot.do("collect", [0.5, -0.5])

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        robot.execute_next_action()
        nav_map.update()
