# Copyright (C) 2021 Jason Brown
#
# SPDX-License-Identifier: MIT
"""Class file for PID controller"""

import matplotlib.pyplot as plt
import numpy as np


class DataRecorder:
    def __init__(self, *args):
        self.dict = {}
        for arg in args:
            self.dict[arg] = []

    def update(self, **kwargs):
        for k in self.dict.keys():
            self.dict[k].append(kwargs[k] if k in kwargs.keys() else 0)

    def reset(self):
        for k in self.dict.keys():
            self.dict[k] = []

    def clear_last(self):
        for k in self.dict.keys():
            self.dict[k] = self.dict[k][:-1]

    def plot(self, x_axis_arg, *args, styles=None, title=None):
        if not self.dict[x_axis_arg]:
            return

        if not args:
            args = list(self.dict.keys())
            args.remove(x_axis_arg)

        for k in args:
            try:
                plt.plot(self.dict[x_axis_arg], self.dict[k], styles[k], label=k)
            except KeyError:
                plt.plot(self.dict[x_axis_arg], self.dict[k], label=k)

        plt.title(title)
        plt.grid()
        plt.legend()
        plt.xlabel(x_axis_arg)
        plt.ylabel("Quanta")
        plt.show()


class PID:
    def __init__(self, quantity, k_p=0, k_i=0, k_d=0, time_step=None, integral_wind_up_speed=np.inf):
        self.quantity = quantity

        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.prev_error = None
        self.cum_error = 0
        self.time_step = time_step

        self.i_wind_up_speed = integral_wind_up_speed
        self.total_time = 0

        self.history = DataRecorder("time", "error", "cumulative_error", "error_change",
                                    "k_p", "k_i", "k_d", "p", "i", "d", "output")

        # For rolling back the pid a step
        self.old_cum_error = 0
        self.old_prev_error = None
        self.old_total_time = 0

    def reset(self):
        self.prev_error = None
        self.cum_error = 0
        self.total_time = 0
        self.old_prev_error = None
        self.old_cum_error = 0
        self.old_total_time = 0

    def step(self, error, time_step=None):
        time_step = time_step if time_step is not None else self.time_step
        if time_step is None:
            raise Exception("No time step provided in step() or init()")

        self.old_cum_error = self.cum_error
        self.old_prev_error = self.prev_error
        self.old_total_time = self.total_time

        self.cum_error += error * time_step * np.tanh(self.total_time * self.i_wind_up_speed)
        first_error_change = (error - self.prev_error) / time_step if self.prev_error is not None else 0
        second_error_change = (self.prev_error - self.old_prev_error) / time_step if self.prev_error is not None and self.old_prev_error is not None else 0
        error_change = 0.5 * first_error_change + 0.5 * second_error_change

        p = self.k_p * error
        i = self.k_i * self.cum_error
        d = self.k_d * error_change

        output = p + i + d

        self.history.update(time=self.total_time, error=error, cumulative_error=self.cum_error,
                            error_change=error_change, k_p=self.k_p, k_i=self.k_i, k_d=self.k_d, p=p, i=i, d=d,
                            output=output)

        self.prev_error = error
        self.total_time += time_step

        return output

    def un_step(self):
        self.cum_error = self.old_cum_error
        self.prev_error = self.old_prev_error
        self.total_time = self.old_total_time
        self.history.clear_last()

    def evaluate(self, *args):
        # Mild deuteranopia go brrrr
        styles = {
            "output": 'k-',
            "error": 'r-',
            "cumulative_error": 'b-',
            "error_change": 'y-',
            "p": 'r--',
            "i": 'b--',
            "d": 'y--',
            "k_p": 'r:',
            "k_i": 'b:',
            "k_d": 'y:',
        }

        title = f"""{self.quantity} PID
{f" K_p={self.k_p} " if self.k_p != 0 else ""}\
{f" K_i={self.k_i} " if self.k_i != 0 else ""}\
{f" K_d={self.k_d} " if self.k_d != 0 else ""}"""

        if not args:
            plot_args = ["output", "error"]
            if self.k_p != 0:
                plot_args.append("p")
            if self.k_i != 0:
                plot_args.extend(["cumulative_error", "i"])
            if self.k_d != 0:
                plot_args.extend(["error_change", "d"])
        else:
            plot_args = args

        self.history.plot("time", *plot_args, styles=styles, title=title)
