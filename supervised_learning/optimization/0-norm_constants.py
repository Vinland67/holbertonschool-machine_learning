#!/usr/bin/env python3
"""
Matris üçün normallaşdırma sabitlərin hesablanması modulu
"""
import numpy as np


def normalization_constants(X):
    """
    Matrisin hər bir xüsusiyyəti üçün orta qiymətini (mean)
    və standart meylini (standard deviation) hesablayır.

    Parametrlər:
        X: (m, nx) ölçülü numpy.ndarray

    Qaytarır:
        mean, std: hər xüsusiyyət üçün sırasıyla orta qiymət və standart meyl
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
