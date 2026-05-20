#!/usr/bin/env python3
"""
Modul DataFrame-in indeksini təyin etmək üçün funksiya təmin edir.
"""


def index(df):
    """
    'Timestamp' sütununu DataFrame-in indeksi olaraq təyin edir.
    """
    return df.set_index('Timestamp')
