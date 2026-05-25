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
    # 1. Hər iki cədvəlin Timestamp sütununu indeks edirik
    df1 = index(df1)
    df2 = index(df2)
    
    # 2. df2 cədvəlini 1417411920 daxil olmaqla kəsirik
    df2 = df2.loc[:1417411920]
    
    # 3. df2 üstə, df1 isə altda olmaqla birləşdirib açar sözlər təyin edirik
    result = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    
    return result
