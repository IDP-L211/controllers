def main(robot):
    action_queue = [
        ("rotate", 6.28 * 2, 2)
    ]

    while robot.step(robot.timestep) != -1:
        robot.execute_action(action_queue)

        # Amazing the amount of effort I put into something I have no intention
        # of ever running...
        print(*map(lambda x: f"{x:5f}" if not isinstance(x, str) else x, (
            "ultrasonic_left value: ",
             robot.ultrasonic_left.getValue(), "\n",
            "ultrasonic_left bounds: ",
            *robot.ultrasonic_left.getBounds(), "\n\n",

            "ultrasonic_right value: ",
             robot.ultrasonic_right.getValue(), "\n",
            "ultrasonic_right bounds: ",
            *robot.ultrasonic_right.getBounds(),

        )), sep="")
