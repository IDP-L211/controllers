# Copyright (C) 2021 Jason Brown
#
# SPDX-License-Identifier: MIT
"""Class file for PID controller"""

import matplotlib.pyplot as plt
import numpy as np


class DataRecorder:
    """Class used to help record PID and motor behaviour

    Attributes:
        dict (dict): The dictionary of lists that contain the data
    """

    def __init__(self, *args, styles=None):
        self.dict = {}
        self.styles = styles
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

    def plot(self, x_axis_arg, *args, title=None):
        if not args:
            args = list(self.dict.keys())
            args.remove(x_axis_arg)

        if not self.dict[x_axis_arg] or np.isnan(self.dict[args[0]]).all():
            return

        for k in args:
            if self.styles is not None:
                plt.plot(self.dict[x_axis_arg], self.dict[k], self.styles[k], label=k)
            else:
                plt.plot(self.dict[x_axis_arg], self.dict[k], label=k)

        plt.title(title)
        plt.grid()
        plt.legend()
        plt.xlabel(x_axis_arg)
        plt.ylabel("Quanta")
        plt.show()


class PID:
    def __init__(self, quantity, timer_func, k_p=0, k_i=0, k_d=0, time_step=None, integral_wind_up_speed=None):
        self.quantity = quantity

        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.prev_error = None
        self.cum_error = 0

        self.i_wind_up_speed = integral_wind_up_speed

        self.time_step = time_step
        self.timer_func = timer_func
        self.active_time = 0
        self.last_time_called = 0

        # Mild deuteranopia go brrrr
        pid_graph_styles = {"output": 'k-', "error": 'r-', "cumulative_error": 'b-', "error_change": 'y-', "p": 'r--',
                            "i": 'b--', "d": 'y--'}
        self.history = DataRecorder("time", "error", "cumulative_error", "error_change", "p", "i", "d", "output",
                                    styles=pid_graph_styles)

        # For rolling back the pid a step
        self.old_cum_error = 0
        self.old_prev_error = None
        self.old_last_time_called = 0

    def reset(self):
        self.prev_error = None
        self.cum_error = 0
        self.old_prev_error = None
        self.old_cum_error = 0
        self.active_time = 0
        self.last_time_called = -1
        self.old_last_time_called = 0

    def step(self, error):
        self.old_cum_error = self.cum_error
        self.old_prev_error = self.prev_error
        time = self.timer_func()

        # Check if there was a gap in being called
        if self.last_time_called + self.time_step < time:
            self.history.update(time=self.last_time_called, error=np.nan, cumulative_error=np.nan, error_change=np.nan,
                                p=np.nan, i=np.nan, d=np.nan, output=np.nan)
        self.old_last_time_called = self.last_time_called
        self.last_time_called = time

        i_windup_term = 1 if self.i_wind_up_speed is None else np.tanh(self.active_time * self.i_wind_up_speed)
        self.cum_error += error * self.time_step * i_windup_term

        # This reduces the chance the derivative terms gets high frequency oscillations
        first_error_change = (error - self.prev_error) / self.time_step if self.prev_error is not None else 0
        second_error_change = (self.prev_error - self.old_prev_error) / self.time_step \
            if self.prev_error is not None and self.old_prev_error is not None else 0
        error_change = 0.5 * first_error_change + 0.5 * second_error_change

        p = self.k_p * error
        i = self.k_i * self.cum_error
        d = self.k_d * error_change

        output = p + i + d

        self.history.update(time=time, error=error, cumulative_error=self.cum_error, error_change=error_change, p=p,
                            i=i, d=d, output=output)

        self.prev_error = error
        self.active_time += self.time_step

        return output

    def un_step(self):
        self.cum_error = self.old_cum_error
        self.prev_error = self.old_prev_error
        self.history.clear_last()
        self.active_time -= self.time_step
        self.last_time_called = self.old_last_time_called

    def plot_history(self, *args):
        title = f"""{self.quantity} PID
{f" K_p={self.k_p} " if self.k_p != 0 else ""}\
{f" K_i={self.k_i} " if self.k_i != 0 else ""}\
{f" K_d={self.k_d} " if self.k_d != 0 else ""}\
{f" i_windup={self.i_wind_up_speed} " if self.i_wind_up_speed is not None else ""}"""

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

        self.history.plot("time", *plot_args, title=title)
