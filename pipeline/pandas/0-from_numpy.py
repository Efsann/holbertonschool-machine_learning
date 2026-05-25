#!/usr/bin/env python3
"""
Defines a function that creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd
import string


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray with alphabetical column labels.
    """
    # Massivin sütun sayını tapırıq
    num_cols = array.shape[1]
    
    # Sütun sayı qədər böyük əlifba hərfi seçirik (A, B, C...)
    col_names = list(string.ascii_uppercase[:num_cols])
    
    # DataFrame yaradırıq
    df = pd.DataFrame(array, columns=col_names)
    
    return df
