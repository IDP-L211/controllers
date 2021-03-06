# Copyright (C) 2021 Weixuan Zhang, Eleanor Clifford, Jason Brown
#
# SPDX-License-Identifier: MIT
"""This module contains a class representing a supervisor - a combination of robot and a human operator.
"""

from controller import Supervisor
from robot import IDPRobot


class IDPSupervisor(IDPRobot, Supervisor):
    def __init__(self):
        super().__init__()
