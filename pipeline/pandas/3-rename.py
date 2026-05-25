#!/usr/bin/env python3
"""
Defines a function that renames a column and converts its values to datetime
"""
import pandas as pd


def rename(df):
    """
    Renames Timestamp column to Datetime, converts it to datetime values,
    and returns only Datetime and Close columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
