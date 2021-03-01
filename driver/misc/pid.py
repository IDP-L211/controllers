# Copyright (C) 2021 Jason Brown
#
# SPDX-License-Identifier: MIT
"""Class file for PID controller"""


import matplotlib.pyplot as plt
import numpy as np


class PID:
    def __init__(self, k_p=0, k_i=0, k_d=0, time_step=None):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.prev_error = None
        self.cum_error = 0
        self.time_step = time_step

        # Each history item will be a dict of quanta
        self.history = []

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

        output = p + i + d

        self.history.append({
            "timestep": time_step,
            "error": error,
            "cumulative_error": self.cum_error,
            "error_change": error_change,
            "k_p": self.k_p,
            "k_i": self.k_i,
            "k_d": self.k_d,
            "p": p,
            "i": i,
            "d": d,
            "output": output
        })

        self.prev_error = error

        return output

    def plot(self, *args):
        x_axis = [0]
        y_axis_quanta = {}

        # From the given args, assemble lists of data to graph
        for entry in self.history:
            x_axis.append(x_axis[-1] + entry["timestep"])
            for arg in args:
                if arg not in y_axis_quanta.keys():
                    y_axis_quanta[arg] = []
                y_axis_quanta[arg].append(entry[arg])

        x_axis = x_axis[:-1]

        # Plot
        for arg in args:
            plt.plot(x_axis, y_axis_quanta[arg], label=arg)

        plt.grid()
        plt.show()
