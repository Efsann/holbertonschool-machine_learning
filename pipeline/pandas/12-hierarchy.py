#!/usr/bin/env python3
"""
Defines a function to rearrange MultiIndex hierarchy
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates two DataFrames, swaps index levels and sorts chronologically
    """
    # 1. Cədvəlləri indeksləyir və göstərilən aralıqda kəsirik
    df1 = index(df1).loc[1417411980:1417417980]
    df2 = index(df2).loc[1417411980:1417417980]
    
    # 2. Cədvəlləri alt-alta birləşdirir və açar sözlər təyin edirik
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    
    # 3. İndeks səviyyələrinin yerini dəyişirik (Timestamp birinci olur)
    df = df.swaplevel(0, 1)
    
    # 4. İndeksə (xronoloji ardıcıllığa) görə çeşidləyirik
    df = df.sort_index()
    
    return df
