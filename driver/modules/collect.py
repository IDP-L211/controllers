from enum import Enum

import numpy as np

from .utils import print_if_debug

tau = np.pi * 2


class CollectStates(Enum):
    APPROACHING_TARGET_FROM_CENTER = 0
    DRIVING_TO_TARGET = 1
    ROTATE_TO_FACE_TARGET = 2
    DETECTING_COLOUR = 3
    CORRECT_COLOUR = 4
    COLLECTING_TARGET = 5
    TARGET_COLLECTED = 6


class CollectHandler:
    def __init__(self, robot, debug_flag=False):
        self.robot = robot

        # PLEASE NOTE! Since the distance accuracy is 0.02, the bot will stop ~0.02 distance from its goal
        # If the goal is within 0.02 it won't move at all as it's 'already there'
        self.colour_detect_distance_start = 0.16
        self.colour_detect_distance_end = 0.14
        self.max_angle_to_block = 0.12
        self.rotate_angle = -tau / 2.5
        self.gate_time = 0.5
        self.reverse_distance = 0.2
        self.collect_rotation_rate = 2.0

        self.debug_flag = debug_flag

        self.collect_state = CollectStates.APPROACHING_TARGET_FROM_CENTER
        self.collect_num_tries = 0
        self.collect_target_pos_cache = None
        self.collect_color_reading = None

    def reset_collect_state(self):
        self.collect_state = CollectStates.APPROACHING_TARGET_FROM_CENTER

    def move_to_block_with_offset(self, offset):
        # First check we haven't already cached the target true position
        if self.robot.collect_handler.collect_target_pos_cache is None:
            self.robot.collect_handler.collect_target_pos_cache = self.robot.target.position

        # Set the target position to where we actually want to go
        self.robot.target.position = self.robot.coordtransform_bot_polar_to_world(
            self.robot.distance_from_bot(self.collect_target_pos_cache) - offset,
            self.robot.angle_from_bot_from_position(self.collect_target_pos_cache))

        # If we completed motion restore target position to true position and empty cache
        if complete := self.robot.drive_to_position(self.robot.target.position, passive_collision_avoidance=False,
                                                    accuracy_threshold=0.02):
            self.robot.target.position = self.collect_target_pos_cache
            self.collect_target_pos_cache = None
        return complete

    def collect(self):
        """Collect block at the best position possible

        Returns:
            bool: If the collect action is completed
        """
        if any((
                not self.robot.target,  # no target selected
                all((
                        self.collect_state in [CollectStates.DRIVING_TO_TARGET,
                                               CollectStates.APPROACHING_TARGET_FROM_CENTER],  # on the way to target
                        not self.robot.check_target_valid(self.robot.target),  # collision if continue to target
                        other_bot_pos := self.robot.radio.get_other_bot_position(),
                        # have position data of the other robot
                        abs(self.robot.angle_from_bot_from_position(other_bot_pos)) < np.pi / 2
                        # the other robot is in front of us
                ))
        )):
            if curr_best_target := self.robot.get_best_target():
                self.robot.target = curr_best_target  # update target to the best available at this time
            else:  # no other target can be selected either, return to scan
                print_if_debug(f"{self.robot.color}, collect: No target can be selected", debug_flag=self.debug_flag)
                self.robot.brake()
                return True

        # Ifs not elifs means we don't waste timesteps if the state changes
        if self.collect_state == CollectStates.APPROACHING_TARGET_FROM_CENTER:  # Approach wide if at edge
            new_target_pos = [0.75 * np.sign(x) if abs(x) > 1.0 else x for x in self.robot.target.position]
            if self.collect_target_pos_cache is None:
                if any(x != y for x, y in zip(new_target_pos, self.robot.target.position)):
                    self.collect_target_pos_cache = self.robot.target.position
                    self.robot.target.position = new_target_pos
                    print_if_debug(f"{self.robot.color}, collect: Block is near edge, driving nearby",
                                   debug_flag=self.debug_flag)
                else:
                    self.collect_state = CollectStates.DRIVING_TO_TARGET
            else:
                if self.robot.drive_to_position(self.robot.target.position, passive_collision_avoidance=False):
                    self.robot.target.position = self.collect_target_pos_cache
                    self.collect_target_pos_cache = None
                    print_if_debug(f"{self.robot.color}, collect: At approach, driving to target",
                                   debug_flag=self.debug_flag)
                    self.collect_state = CollectStates.DRIVING_TO_TARGET

        if self.collect_state == CollectStates.DRIVING_TO_TARGET:  # driving to target
            if self.move_to_block_with_offset(self.colour_detect_distance_start):
                print_if_debug(f"{self.robot.color}, collect: At target, rotating", debug_flag=self.debug_flag)
                self.collect_state = CollectStates.ROTATE_TO_FACE_TARGET

        if self.collect_state == CollectStates.ROTATE_TO_FACE_TARGET:
            angle_to_block = self.robot.angle_from_bot_from_position(self.robot.target.position)
            if abs(angle_to_block) > self.max_angle_to_block:
                self.robot.rotate(angle_to_block)
            else:
                self.robot.stored_time = self.robot.time
                print_if_debug(f"{self.robot.color}, collect: Facing target, detecting color",
                               debug_flag=self.debug_flag)
                self.collect_state = CollectStates.DETECTING_COLOUR

        if self.collect_state == CollectStates.DETECTING_COLOUR:
            detection = self.robot.color_detector.get_color()
            if detection in ["red", "green"] or self.collect_color_reading not in ["red", "green"]:
                self.collect_color_reading = detection
            if self.move_to_block_with_offset(self.colour_detect_distance_end):
                print(f"Block colour: {self.collect_color_reading}s")
                if self.collect_color_reading in ["red", "green"]:
                    self.robot.target.classification = f"{self.collect_color_reading}_box"
                    if self.collect_color_reading == self.robot.color:
                        print_if_debug(f"{self.robot.color}, collect: Color match, collecting",
                                       debug_flag=self.debug_flag)
                        self.collect_state = CollectStates.CORRECT_COLOUR
                    else:
                        self.robot.action_queue.insert(1, (
                            "reverse", list(self.robot.get_bot_front(-self.reverse_distance))))
                        # should be collected by the other robot, send info to it
                        self.robot.radio.send_message(
                            {'confirmed': (list(self.robot.target.position), self.collect_color_reading)})
                        self.robot.target = None
                        print_if_debug(f"{self.robot.color}, collect: Color opposite, reversing",
                                       debug_flag=self.debug_flag)
                        return True
                elif (other_bot_collected := self.robot.radio.get_other_bot_collected()) and len(
                        other_bot_collected) == 4 \
                        and self.robot.target.classification == f'{self.robot.color}_box':
                    print_if_debug(f"{self.robot.color}, collect: Must be ours, collecting",
                                   debug_flag=self.debug_flag)
                    # other robot has collected all four targets, this must be ours
                    self.collect_state = CollectStates.CORRECT_COLOUR
                else:  # not able to detect the colour, probably because the position is not accurate
                    self.robot.action_queue.insert(1,
                                                   ("reverse", list(self.robot.get_bot_front(-self.reverse_distance))))
                    self.robot.action_queue.insert(2, "scan")  # rescan to check if there is actually a target there
                    # pop the current target off for now, the new scan will give better position
                    if self.collect_num_tries > 1:  # this block is probably flipped over
                        self.robot.target.classification = 'flipped'
                        self.collect_num_tries = 0  # reset the counter
                        print_if_debug(f"{self.robot.color}, collect: Color unknown, I think this is flipped",
                                       debug_flag=self.debug_flag)
                    else:
                        self.robot.target_cache.remove_target(self.robot.target)
                        self.collect_num_tries += 1
                        print_if_debug(f"{self.robot.color}, collect: Color unknown, I'll try again later",
                                       debug_flag=self.debug_flag)
                    self.robot.target = None
                    return True

        if self.collect_state == CollectStates.CORRECT_COLOUR:
            self.robot.gate.open()
            if self.robot.time - self.robot.stored_time >= self.gate_time:
                print_if_debug(f"{self.robot.color}, collect: Gate open, rotating", debug_flag=self.debug_flag)
                self.collect_state = CollectStates.COLLECTING_TARGET

        if self.collect_state == CollectStates.COLLECTING_TARGET:
            if self.robot.rotate(self.rotate_angle, max_rotation_rate=self.collect_rotation_rate):
                self.robot.stored_time = self.robot.time
                print_if_debug(f"{self.robot.color}, collect: Collected, closing gate", debug_flag=self.debug_flag)
                self.collect_state = CollectStates.TARGET_COLLECTED

        if self.collect_state == CollectStates.TARGET_COLLECTED:
            self.robot.gate.close()
            if self.robot.time - self.robot.stored_time >= self.gate_time:
                self.robot.target_cache.remove_target(self.robot.target)
                self.robot.target_cache.collected.append(self.robot.target)
                self.robot.target = None
                print_if_debug(f"{self.robot.color}, collect: Gate closed, moving on", debug_flag=self.debug_flag)
                return True

        return False
