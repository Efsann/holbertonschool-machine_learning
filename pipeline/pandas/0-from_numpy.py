#!/usr/bin/env python3
"""
Defines a function that creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray with alphabetical column labels.

    Args:
        array (np.ndarray): The numpy array to convert.

    Returns:
        pd.DataFrame: The newly created DataFrame.
    """
    # Massivin neçə sütundan ibarət olduğunu tapırıq
    num_cols = array.shape[1]
    
    # Sütun sayı qədər əlifba hərfi generasiya edirik (A, B, C...)
    # chr(65) = 'A', chr(66) = 'B' və s.
    col_names = [chr(65 + i) for i in range(num_cols)]
    
    # DataFrame yaradırıq və sütun adlarını təyin edirik
    df = pd.DataFrame(array, columns=col_names)
    
    return df
