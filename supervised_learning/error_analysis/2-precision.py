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
    # Diaqonaldakı True Positive-ləri götürürük
    tp = np.diag(confusion)
    
    # Sütun üzrə cəmləri (True Positives + False Positives) tapırıq
    predicted_total = np.sum(confusion, axis=0)
    
    # Precision = TP / (TP + FP)
    return tp / predicted_total
