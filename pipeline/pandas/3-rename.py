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
    # 1. Sütunun adını dəyişirik
    df = df.rename(columns={'Timestamp': 'Datetime'})
    
    # 2. UNIX saniyələrini oxunabilən tarix formatına çeviririk
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    
    # 3. Yalnız Datetime və Close sütunlarını seçirik
    df = df[['Datetime', 'Close']]
    
    return df
