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

    def reset(self):
        for k in self.dict.keys():
            self.dict[k] = []

    def update(self, **kwargs):
        for k in self.dict.keys():
            self.dict[k].append(kwargs[k] if k in kwargs.keys() else 0)

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
    def __init__(self, k_p=0, k_i=0, k_d=0, time_step=None, integral_wind_up_speed=np.inf, integral_delay_time=0,
                 integral_active_error_band=np.inf, derivative_weight_decay_half_life=None, custom_function=None,
                 integral_delay_windup_when_in_bounds=True, quantity_name="Error", timer_func=None):
        self.quantity_name = quantity_name

        if custom_function is not None and (k_p != 0 or k_i != 0 or k_d != 0):
            raise Exception("Both constants and custom function specified, use one or the other")

        # Log these so we can display them in the graph title
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        # If custom, this should take 3 arguments (error, cumulative_error, error_change) and return one output
        self.custom_function = custom_function

        # This helps stabilise the derivative error by using an exponential moving average of error changes
        self.d_weight_decay_half_life = derivative_weight_decay_half_life
        if derivative_weight_decay_half_life is not None:
            num_weights = int(3.32 * derivative_weight_decay_half_life / time_step)  # Last weight is 10% of first
            weight_decay_constant = derivative_weight_decay_half_life / time_step
            raw_weights = np.array([2 ** (-i / weight_decay_constant) for i in range(num_weights)])
            self.e_weights = raw_weights / sum(raw_weights)
        else:
            self.e_weights = [1]
        self.e_history = []  # Record the last set of errors for derivative calc

        self.cum_error = 0
        self.i_wind_up_speed = integral_wind_up_speed
        self.i_delay_time = integral_delay_time
        self.i_active_error_band = integral_active_error_band
        self.i_delay_windup_when_in_bounds = integral_delay_windup_when_in_bounds

        self.time_step = time_step
        self.timer_func = timer_func
        self.active_time = 0
        self.last_time_called = 0
        self.time_entered_bounds = None

        # Mild deuteranopia go brrrr
        pid_graph_styles = {"output": 'k-', "error": 'r-', "cumulative_error": 'b-', "error_change": 'y-', "p": 'r--',
                            "i": 'b--', "d": 'y--'}
        self.history = DataRecorder("time", "error", "cumulative_error", "error_change", "p", "i", "d", "output",
                                    styles=pid_graph_styles)

    def reset(self, hard=False):
        self.e_history = []
        self.cum_error = 0
        self.active_time = 0 if self.timer_func is not None else self.active_time
        self.time_entered_bounds = None
        if hard:
            self.history.reset()

    def query(self, error, step_mode=False):
        """Get the output of the pid without affecting its state"""
        time = self.timer_func() if self.timer_func is not None else self.active_time

        # Check if there was a gap in being called, if so make sure the logs and graphs reflect this
        # Whilst resetting here might go against the idea of query, it would get reset anyway if we waited for step
        # Therefore we reset here so the returned result is accurate even if not stepping
        if self.last_time_called + self.time_step < time:
            if step_mode:
                self.history.update(time=self.last_time_called, error=np.nan, cumulative_error=np.nan, error_change=np.nan,
                                    p=np.nan, i=np.nan, d=np.nan, output=np.nan)
            self.reset()

        error_change = sum([(e1 - e2) * w / self.time_step
                            for e1, e2, w in zip(self.e_history[::-1], self.e_history[-2::-1], self.e_weights)])
        if self.custom_function is not None:
            p, i, d = np.nan, np.nan, np.nan
            output = self.custom_function(error, self.cum_error, error_change)
        else:
            p = self.k_p * error
            i = self.k_i * self.cum_error
            d = self.k_d * error_change
            output = p + i + d

        if step_mode:
            return output, p, i, d, error_change, time
        else:
            return output

    def step(self, error):
        """Get the output and move the controller through a time step, updating logs"""
        output, p, i, d, error_change, time = self.query(error, True)
        self.last_time_called = time

        self.history.update(time=time, error=error, cumulative_error=self.cum_error, error_change=error_change, p=p,
                            i=i, d=d, output=output)

        # If we are waiting to be in our bounds before applying delay and windup we need the to use the correct time
        if self.i_delay_windup_when_in_bounds:
            if abs(error) <= self.i_active_error_band:
                if self.time_entered_bounds is None:
                    self.time_entered_bounds = self.active_time
            else:
                self.time_entered_bounds = None
            i_time = self.active_time - self.time_entered_bounds if self.time_entered_bounds is not None else 0
        else:
            i_time = self.active_time

        # Use delay and windup to calculate the weight on error accumulation
        i_control_term = max(0, np.tanh((i_time - self.i_delay_time) * self.i_wind_up_speed))
        i_control_term = i_control_term if abs(error) <= self.i_active_error_band else 0

        # Update our state variables
        self.cum_error += error * self.time_step * i_control_term
        self.e_history.append(error)
        self.active_time += self.time_step

        return output

    def plot_history(self, *args):
        title = f"""{self.quantity_name} PID\n{f" K_p={self.k_p} " if self.k_p != 0 else ""}\
{f" K_i={self.k_i} " if self.k_i != 0 else ""}{f" K_d={self.k_d} " if self.k_d != 0 else ""}\
{f"custom_function={self.custom_function.__name__}" if self.custom_function is not None else ""}\
{f" i_windup={self.i_wind_up_speed} " if self.i_wind_up_speed != np.inf else ""}\
{f" i_delay={self.i_delay_time} " if self.i_delay_time != 0 else ""}\
{f" i_active_error_band={self.i_active_error_band} " if self.i_active_error_band != np.inf else ""}\
{f" d_weight_decay_half_life={self.d_weight_decay_half_life} " if self.d_weight_decay_half_life is not None else ""}"""

        if not args:
            if self.custom_function is None:
                plot_args = ["output", "error"] + (["p"] * bool(self.k_p)) + (["cumulative_error", "i"] * bool(self.k_i))\
                    + (["error_change", "d"] * bool(self.k_d))  # Now this is pod-racing
            else:
                plot_args = ["output", "error", "cumulative_error", "error_change"]
        else:
            plot_args = args

        self.history.plot("time", *plot_args, title=title)
