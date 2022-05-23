#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : TODO
@File      : timing.py
@Project   : BrickScanner
@Time      : 05.05.22 16:13
@Author    : flowmeadow
"""
import time
from typing import Callable, Tuple

import numpy as np


def time_fun(fun: Callable, args=None, kwargs=None, repeat=1) -> Tuple[float, object]:
    """
    :param fun: function to execute
    :param repeat: number of executions
    :param args: function arguments
    :param kwargs: function keyword arguments
    :return: mean execution time, function return
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = dict()
    result = None
    times = []
    for i in range(repeat):
        start = time.perf_counter()  # start timer
        result = fun(*args, **kwargs)  # execute
        t = time.perf_counter() - start  # stop timer
        times.append(t)
    t_mean = np.mean(np.array(times))
    return t_mean, result


if __name__ == "__main__":
    pass
