# Test script for drive_to_pos

def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())

    # Actions for our robot
    action_queue = [
        ("move", [0, 0]),
        "scan",
        ("move", [0.9, 0.9]),
        ("move", [0.9, -0.9]),
        ("move", [-0.9, 0.9]),
        ("move", [-0.9, -0.9]),
        ("move", [0, 0])
    ]

    robot.action_queue = action_queue

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        robot.execute_next_action()
