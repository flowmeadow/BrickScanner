#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce : Contains timing methods
@File      : timing.py
@Project   : BrickScanner
@Time      : 05.05.22 16:13
@Author    : flowmeadow
"""
import time
from typing import Callable, Tuple, Optional

import numpy as np


def time_fun(
    fun: Callable,
    args: Optional[list] = None,
    kwargs: Optional[dict] = None,
    repeat: int = 1,
) -> Tuple[float, object]:
    """
    Method to time a given function several times to get the execution time average
    :param fun: function to execute
    :param args: function arguments
    :param kwargs: function keyword arguments
    :param repeat: number of executions
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
