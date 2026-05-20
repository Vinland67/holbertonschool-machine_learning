#!/usr/bin/env python3
"""
Modul DataFrame-i müəyyən sütuna görə sıralamaq üçün funksiya təmin edir.
"""


def high(df):
    """
    DataFrame-i 'High' sütununa görə azalan sırada çeşidləyir.
    """
    return df.sort_values(by='High', ascending=False)
