#!/usr/bin/env python3
"""
Hər sinif üçün F1 score hesablayan modul
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    confusion: (classes, classes) ölçülü confusion matrix
    Returns: hər sinif üçün F1 score dəyərlərini saxlayan (classes,) massivi
    """
    p = precision(confusion)
    r = sensitivity(confusion)
    return 2 * (p * r) / (p + r)
