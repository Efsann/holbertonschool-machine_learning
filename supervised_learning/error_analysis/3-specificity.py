#!/usr/bin/env python3
"""
Hər sinif üçün specificity hesablayan modul
"""
import numpy as np


def specificity(confusion):
    """
    confusion: (classes, classes) ölçülü confusion matrix
    Returns: hər sinif üçün specificity dəyərlərini saxlayan (classes,) massivi
    """
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    total = np.sum(confusion)
    
    tn = total - (tp + fp + fn)
    return tn / (tn + fp)
