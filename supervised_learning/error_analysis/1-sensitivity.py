#!/usr/bin/env python3
"""
Hər sinif üçün sensitivity (recall) hesablayan modul
"""
import numpy as np


def sensitivity(confusion):
    """
    confusion: (classes, classes) ölçülü confusion matrix
    Returns: hər sinif üçün sensitivity dəyərlərini saxlayan (classes,) massivi
    """
    tp = np.diag(confusion)
    actual_total = np.sum(confusion, axis=1)
    return tp / actual_total
