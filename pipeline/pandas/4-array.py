#!/usr/bin/env python3
"""
Defines a function that selects data from a pd.DataFrame and converts to numpy
"""
import pandas as pd


def array(df):
    """
    Selects the last 10 rows of High and Close columns and converts to numpy.
    """
    selected_data = df[['High', 'Close']].tail(10).to_numpy()
    return selected_data
