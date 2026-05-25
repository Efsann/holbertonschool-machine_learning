#!/usr/bin/env python3
"""
Defines a function that loads data from a file as a pd.DataFrame
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame.

    Args:
        filename (str): The path to the file to load.
        delimiter (str): The column separator.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(filename, sep=delimiter)
    return df
