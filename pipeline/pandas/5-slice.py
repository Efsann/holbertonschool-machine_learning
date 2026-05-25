#!/usr/bin/env python3
"""
Defines a function to slice a DataFrame
"""


def slice(df):
    """
    Extracts specific columns and selects every 60th row
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']][::60]
