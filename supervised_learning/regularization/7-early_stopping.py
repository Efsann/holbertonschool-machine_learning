#!/usr/bin/env python3
"""
Module to determine early stopping for gradient descent
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should stop early.

    Parameters:
    cost: current validation cost
    opt_cost: lowest recorded validation cost
    threshold: threshold used for early stopping
    patience: patience count used for early stopping
    count: count of how long the threshold has not been met

    Returns:
    A boolean indicating whether the network should stop early,
    followed by the updated count
    """
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1

    boolean = count >= patience

    return boolean, count
