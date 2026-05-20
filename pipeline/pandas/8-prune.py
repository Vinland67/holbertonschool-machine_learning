#!/usr/bin/env python3
"""
Modul DataFrame-dən NaN dəyərlərinin silinməsi üçün funksiya təmin edir.
"""


def prune(df):
    """
    'Close' sütununda NaN dəyəri olan bütün sətirləri DataFrame-dən silir.
    """
    return df.dropna(subset=['Close'])
