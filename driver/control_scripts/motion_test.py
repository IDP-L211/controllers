# Test script for drive_to_pos

def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())
    nav_map = robot.get_map(robot.ir_long, 2.4)

    # Actions for our robot
    action_queue = [
        ("move", [1, -1]),
        ("move", [1, 1]),
        ("face", 3.141),
        ("move", [-1, 1]),
        ("move", [-1, -1]),
        ("move", [0, 0]),
        ("face", 0.0)
    ]

    # Main loop, perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        robot.execute_action(action_queue)
        print(robot.position, robot.speed, robot.bearing)
        nav_map.update()
