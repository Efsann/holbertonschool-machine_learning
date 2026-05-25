#!/usr/bin/env python3
"""
Defines a function to remove NaN values from a specific column
"""


def prune(df):
    """
    Removes entries where Close has NaN values
    """
    return df.dropna(subset=['Close'])
