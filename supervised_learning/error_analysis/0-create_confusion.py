#!/usr/bin/env python3
"""
Confusion matrix yaradan modul
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Real və proqnoz edilən etiketlər əsasında confusion matrix qurur.

    Parametrlər:
    labels: (m, classes) ölçülü real siniflərin one-hot massivi
    logits: (m, classes) ölçülü təxminlərin one-hot massivi

    Geri qaytarır:
    (classes, classes) ölçülü confusion matrix massivi
    """
    return np.dot(labels.T, logits)
