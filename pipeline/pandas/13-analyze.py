#!/usr/bin/env python3
"""
Modul DataFrame sütunlarının statistikasını analiz edir.
"""


def analyze(df):
    """
    Timestamp xaric bütün sütunların təsviri statistikasını qaytarır.
    """
    return df.drop(columns=['Timestamp']).describe()
