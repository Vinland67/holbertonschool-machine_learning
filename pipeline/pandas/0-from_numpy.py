#!/usr/bin/env python3
"""
Modul NumPy massivindən Pandas DataFrame yaradılmasını təmin edir.
"""
import pandas as pd


def from_numpy(array):
    """
    NumPy massivindən Pandas DataFrame yaradır.

    Sütunlar əlifba sırası ilə böyük hərflərlə (A, B, C...) adlandırılır.
    Maksimum 26 sütun dəstəklənir.
    """
    num_cols = array.shape[1]
    columns = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=columns)
