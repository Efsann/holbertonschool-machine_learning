#!/usr/bin/env python3
"""
Hər sinif üçün precision hesablayan modul
"""
import numpy as np


def precision(confusion):
    """
    confusion: (classes, classes) ölçülü confusion matrix
    Returns: hər sinif üçün precision dəyərlərini saxlayan (classes,) massivi
    """
    tp = np.diag(confusion)
    predicted_total = np.sum(confusion, axis=0)
    return tp / predicted_total
