#!/usr/bin/env python3
"""
Bu modul fayldan məlumat yükləmək üçün funksiya təmin edir.
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Faylı pd.DataFrame kimi yükləyir.
    """
    return pd.read_csv(filename, sep=delimiter)
