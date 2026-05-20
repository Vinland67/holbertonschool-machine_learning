#!/usr/bin/env python3
"""
Modul DataFrame-dən NumPy massivi yaratmaq üçün funksiya təmin edir.
"""


def array(df):
    """
    High və Close sütunlarının son 10 sətrini numpy.ndarray-ə çevirir.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
