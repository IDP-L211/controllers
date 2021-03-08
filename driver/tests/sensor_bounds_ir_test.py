def main(robot):
    action_queue = [
        ("rotate", 6.28 * 2, 2)
    ]

    while robot.step(robot.timestep) != -1:
        robot.execute_action(action_queue)

        dis = robot.get_sensor_distance_to_wall()
        print(*map(lambda x: f"{x:5f}", (
            robot.infrared.getValue(), *robot.infrared.getBounds()
        )))
        robot.map.plot_coordinate(robot.get_bot_front(dis))
        robot.map.update()
