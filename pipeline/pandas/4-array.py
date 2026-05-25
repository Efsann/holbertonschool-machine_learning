#!/usr/bin/env python3
"""
Defines a function to convert a dataframe to numpy array
"""


def array(df):
    """
    Selects the last 10 rows of High and Close columns
    """
    return df[['High', 'Close']].tail(10).to_numpy()
