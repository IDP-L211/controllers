# Test script for drive_to_pos

def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())
    nav_map = robot.get_map(robot.infrared)

    # Actions for our robot
    action_queue = [
        ("rotate", 7),
        ("move", [0.75, -0.75]),
        ("reverse", [-0.75, -0.75]),
        ("move", [0.75, 0.75]),
        ("face", 3.141),
        ("move", [-0.75, 0.75]),
        ("move", [-0.75, -0.75]),
        ("move", [0, 0]),
        ("face", 0.0),
        ("rotate", -5)
    ]

    robot.action_queue = action_queue

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        robot.execute_next_action()
        nav_map.update()
