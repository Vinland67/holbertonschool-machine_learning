#!/usr/bin/env python3
"""
Modul DataFrame üzərində kəsik (slice) əməliyyatını yerinə yetirir.
"""


def slice(df):
    """
    Müəyyən sütunların hər 60-cı sətrini seçib DataFrame kimi qaytarır.
    """
    columns = ['High', 'Low', 'Close', 'Volume_(BTC)']
    return df[columns].iloc[::60]
