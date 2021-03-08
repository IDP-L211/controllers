# Copyright (C) 2021 Jason Brown
#
# SPDX-License-Identifier: MIT
"""This module contains some helper functions.
"""

from multiprocessing import Process


def ensure_list_or_tuple(item):
    return item if isinstance(item, (list, tuple)) else None if item is None else [item]


def fire_and_forget(function, *args, **kwargs):
    """Execute a function separately whilst carrying on with the rest of the programs execution.
    Function MUST be accessible from main scope (not nested in another function). It can be a class method"""
    process = Process(target=function, args=args, kwargs=kwargs)
    process.start()


def print_if_debug(*args, debug_flag=True):
    if debug_flag:
        print(*args)
