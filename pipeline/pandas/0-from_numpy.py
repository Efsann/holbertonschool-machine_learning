#!/usr/bin/env python3
"""
Defines a function that creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray with alphabetical column labels.
    """
    num_cols = array.shape[1]
    col_names = [chr(65 + i) for i in range(num_cols)]
    df = pd.DataFrame(array, columns=col_names)
    return df
