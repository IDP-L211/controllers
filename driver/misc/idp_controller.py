# Copyright (C) 2021 Jason Brown
#
# SPDX-License-Identifier: MIT
"""Class file for the robot controller"""


import numpy as np


class IDPController:
    def __init__(self, robot):
        """Controller for the robot"""
        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())

        # Function dict for high level action functions
        # High level action functions take some arguments and return a list of low level actions
        self.hl_action_functions = {
            "collect": self.collect_block
        }

    @property
    def idle(self):
        """Whether bot is doing something"""
        return self.robot.action_queue == []

    def step(self):
        """Executes operations that are once per timestep"""
        self.robot.execute_next_action()
        return self.robot.step(self.time_step) != -1

    def do_list(self, actions: list, mode="last", only_if_idle=False):
        """Tell the bot to do an action

        Args:
            actions (list): List of action tuples
            mode (string): Which mode to add action to action_queue
            only_if_idle (bool): Whether to only add action if queue is emtpy
        """

        # Swap high level action functions for their composite functions
        low_level_actions = []
        for action in actions:
            if action[0] in self.hl_action_functions.keys():
                low_level_actions.extend(self.hl_action_functions[action[0]](*action[1:]))
            else:
                low_level_actions.append(action)

        """
        # One day this will be valid, one day...
        actions = [*self.hl_action_functions[action[0]](action[1:])
                   if action[0] in self.hl_action_functions else action
                   for action in actions]
        """

        mode = mode.lower()
        if mode not in ["first", "last", "only"]:
            raise Exception(f"{mode} is not a valid mode. Please use first, last or only.")

        if only_if_idle and not self.idle:
            return

        if mode == "first":
            self.robot.action_queue = low_level_actions + self.robot.action_queue
        elif mode == "last":
            self.robot.action_queue.extend(low_level_actions)
        elif mode == "only":
            self.robot.action_queue = low_level_actions

    def do(self, *args, mode="last", only_if_idle=False):
        """Tell the bot to do an action

        Args:
            *args: The action (As arg is now an action tuple, ty python syntactic sugar)
            mode (string): Which mode to add action to action_queue
            only_if_idle (bool): Whether to only add action if queue is emtpy
        """
        self.do_list([args], mode=mode, only_if_idle=only_if_idle)

    def collect_block(self, block_pos):
        """Collect block at position

        Args:
            block_pos ([float, float]): The East-North co-ords of the blocks position
        Returns:
            bool: If we are at our target
        """

        # Update these variables when we have more info
        distance_from_block_to_stop = 0.1
        rotate_angle = np.pi
        home_pos = [0, 0]

        # Calculate pos to got to to be near block not on it
        distance = self.robot.distance_from_bot(block_pos) - distance_from_block_to_stop
        target_pos = self.robot.coordtransform_bot_polar_to_world(distance,
                                                                  self.robot.angle_from_bot_from_position(block_pos))

        # Need to add action that deposits block
        actions = [
            ("move", target_pos),
            ("rotate", rotate_angle),
            ("move", home_pos)
        ]
        return actions
