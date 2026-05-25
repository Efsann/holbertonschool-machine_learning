#!/usr/bin/env python3
"""
Defines a function to select data
"""
import pandas as pd


def array(df):
    """
    Selects specific rows and columns
    """
    selected_data = df[['High', 'Close']].tail(10).to_numpy()
    return selected_data
