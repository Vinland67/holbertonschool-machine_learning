#!/usr/bin/env python3
"""
Modul sütun adının dəyişdirilməsi və məlumatların filtrasiyası üçündür.
"""
import pandas as pd


def rename(df):
    """
    Timestamp sütununu Datetime edir, vaxta çevirir və iki sütun qaytarır.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
