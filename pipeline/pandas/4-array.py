#!/usr/bin/env python3
"""
Defines array function
"""
import pandas as pd


def array(df):
    """
    Selects rows and columns
    """
    selected_data = df[['High', 'Close']].tail(10).to_numpy()
    return selected_data
