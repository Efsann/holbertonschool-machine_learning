#!/usr/bin/env python3
"""
Defines a function to compute descriptive statistics
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp
    """
    return df.drop(columns=['Timestamp']).describe()
