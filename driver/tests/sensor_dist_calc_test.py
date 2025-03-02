def main(robot):
    action_queue = [
        ("rotate", 6.28 * 2, 2)
    ]
    robot.action_queue = action_queue

    while robot.step(robot.timestep) != -1:
        robot.execute_next_action()

        dis = robot.get_sensor_distance_to_wall()
        print(dis)
        robot.map.plot_coordinate(robot.get_bot_front(dis))
        robot.map.update()
