#!/usr/bin/env python3
"""
Defines a function to set the index of a DataFrame
"""


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe
    """
    return df.set_index('Timestamp')
