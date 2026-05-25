#!/usr/bin/env python3
"""Defines array"""
import pandas as pd


def array(df):
    """Selects"""
    return df[['High', 'Close']].tail(10).to_numpy()
