#!/usr/bin/env python3
"""
Defines a function to fill missing data
"""


def fill(df):
    """
    Fills missing values and removes a column
    """
    # 1. Weighted_Price sütununu silirik
    df = df.drop(columns=['Weighted_Price'])
    
    # 2. Close sütununu bir əvvəlki dəyərlə doldururuq (forward fill)
    df['Close'] = df['Close'].ffill()
    
    # 3. High, Low və Open sütunlarını Close-un dəyəri ilə doldururuq
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])
    
    # 4. Həcm sütunlarını 0 ilə doldururuq
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    
    return df
