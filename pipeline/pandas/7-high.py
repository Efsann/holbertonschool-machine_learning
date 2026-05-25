#!/usr/bin/env python3
"""
Defines a function to sort by High price
"""


def high(df):
    """
    Sorts DataFrame by High price in descending order
    """
    return df.sort_values(by='High', ascending=False)
