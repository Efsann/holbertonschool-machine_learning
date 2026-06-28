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
    # 1. Diaqonaldakı True Positive-ləri götürürük
    tp = np.diag(confusion)
    
    # 2. False Positive-ləri tapırıq (Sütun cəmi - TP)
    fp = np.sum(confusion, axis=0) - tp
    
    # 3. False Negative-ləri tapırıq (Sətir cəmi - TP)
    fn = np.sum(confusion, axis=1) - tp
    
    # 4. Matrisdəki bütün elementlərin ümumi cəmi
    total = np.sum(confusion)
    
    # 5. True Negative-ləri hesablayırıq
    tn = total - (tp + fp + fn)
    
    # Specificity = TN / (TN + FP)
    return tn / (tn + fp)
