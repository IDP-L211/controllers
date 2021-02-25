# Test script for object processing

from numpy.random import rand
from numpy import pi


def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())
    robot.step(timestep)

    distances = rand(5) * 2
    angles = (rand(5) - 0.5) * pi

    example_detections = list(zip(distances, angles))
    print('\n'.join(str(x) for x in example_detections))

    for detection in example_detections:
        robot.log_object_detection(detection[0], detection[1], classification="box")

    print(robot.object_detection_handler.objects)
    object_position_list = robot.object_detection_handler.get_box_positions_list(robot)
    print(object_position_list)
    print([robot.distance_from_bot(x) for x in object_position_list])
