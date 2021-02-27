# Copyright (C) 2021 Jason Brown
#
# SPDX-License-Identifier: MIT
"""Class file for PID controller"""


class PID:
    def __init__(self, k_p=0, k_i=0, k_d=0, time_step=None):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.prev_error = None
        self.cum_error = 0
        self.time_step = time_step

    def reset(self):
        self.prev_error = None
        self.cum_error = 0

    def step(self, error, time_step=None):
        time_step = time_step if time_step is not None else self.time_step
        if time_step is None:
            raise Exception("No time step provided in step() or init()")

        self.cum_error += error * time_step
        error_change = (error - self.prev_error) / time_step if self.prev_error is not None else 0

        p = self.k_p * error
        i = self.k_i * self.cum_error
        d = self.k_d * error_change

        self.prev_error = error

        return p + i + d
