# Copyright (C) 2021 Jason Brown
#
# SPDX-License-Identifier: MIT
"""Class file for PID controller"""


import matplotlib.pyplot as plt
import numpy as np
from .utils import round_to_n


class PID:
    def __init__(self, quantity, k_p=0, k_i=0, k_d=0, time_step=None):
        self.quantity = quantity

        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.prev_error = None
        self.cum_error = 0
        self.time_step = time_step

        # Each history item will be a dict of quanta
        self.history = []

        # For rolling back the pid a step
        self.old_cum_error = 0
        self.old_prev_error = None
        self.old_history = []

    def reset(self):
        self.prev_error = None
        self.cum_error = 0
        self.old_prev_error = None
        self.old_cum_error = 0

    def step(self, error, time_step=None):
        time_step = time_step if time_step is not None else self.time_step
        if time_step is None:
            raise Exception("No time step provided in step() or init()")

        self.old_cum_error = self.cum_error
        self.old_prev_error = self.prev_error
        self.old_history = self.history

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

    def un_step(self):
        self.cum_error = self.old_cum_error
        self.prev_error = self.old_prev_error
        self.history = self.old_history


    def evaluate(self, *args, plt_show=True, settle_value=0, settle_threshold=0, analysis_arg="error",
                 show_settle_threshold=False):
        x_axis = np.array([0])
        y_axis_quanta = {}

        # From the given args, assemble lists of data to graph
        for entry in self.history:
            x_axis = np.append(x_axis, x_axis[-1] + entry["timestep"])
            for arg in args:
                if arg not in y_axis_quanta.keys():
                    y_axis_quanta[arg] = np.array([])
                y_axis_quanta[arg] = np.append(y_axis_quanta[arg], entry[arg])

        x_axis = x_axis[:-1]

        adjusted_values = y_axis_quanta[analysis_arg] - settle_value
        max_overshoot = round_to_n(min(adjusted_values) if adjusted_values[0] > 0 else max(adjusted_values), 3)

        settled_time = 0
        for i, t in enumerate(x_axis[::-1]):
            if abs(adjusted_values[-(i+1)]) <= settle_threshold:
                settled_time = t
            else:
                break
        settle_time = round_to_n(x_axis[-1] - settled_time, 3)

        if plt_show:
            # Plot
            for arg in args:
                plt.plot(x_axis, y_axis_quanta[arg], label=arg)

            if show_settle_threshold:
                plt.axhline(y=settle_value + settle_threshold, color='b', linestyle='--')
                plt.axhline(y=settle_value - settle_threshold, color='b', linestyle='--')

            plt.title(f"""{self.quantity} PID
{f" K_p={self.k_p} " if self.k_p != 0 else ""}\
{f" K_i={self.k_i} " if self.k_i != 0 else ""}\
{f" K_d={self.k_d} " if self.k_d != 0 else ""}
{settle_time=}s, {max_overshoot=}""")

            plt.grid()
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Quanta")
            plt.show()

        return max_overshoot, settle_time
