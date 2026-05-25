#!/usr/bin/env python3
"""
Defines a function to concatenate two DataFrames
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes both dataframes, filters df2, and concatenates them
    """
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc[:1417411920]
    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
