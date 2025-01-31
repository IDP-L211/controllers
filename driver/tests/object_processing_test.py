# Test script for object processing

from numpy.random import rand
from numpy import pi


def main(robot):
    # Setup
    timestep = int(robot.getBasicTimeStep())
    robot.step(timestep)

    x_coords = (rand(5) - 0.5) * 2
    z_coords = (rand(5) - 0.5) * 2

    example_detections = list(zip(x_coords, z_coords))
    print('\n'.join(str(x) for x in example_detections))
    print()

    for detection in example_detections:
        robot.log_object_detection(detection, classification="box")

    print(robot.target_cache.targets)
    print()
    object_list = robot.target_cache.get_targets(valid_classes=["box"],
                                                 key=lambda v: robot.distance_from_bot(v.position))
    object_position_list = [d.position for d in object_list]
    print(object_position_list)
    print()
    print([robot.distance_from_bot(x) for x in object_position_list])
