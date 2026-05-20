#!/usr/bin/env python3
"""
Modul MultiIndex strukturunu dəyişib iyerarxiya qurmaq üçün funksiya təmin edir.
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Dataframe-ləri birləşdirir, Timestamp-i ilk səviyyə edir və sıralayır.
    """
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Lazımi zaman aralığını (1417411980 - 1417417980) filtrləyirik
    start, end = 1417411980, 1417417980
    df1_filtered = df1_indexed.loc[start:end]
    df2_filtered = df2_indexed.loc[start:end]

    # DataFrame-ləri etiketlər ilə birləşdiririk
    df = pd.concat(
        [df2_filtered, df1_filtered],
        keys=['bitstamp', 'coinbase']
    )

    # İndeks səviyyələrinin yerini dəyişirik (Timestamp ilk səviyyə olur)
    df = df.swaplevel(0, 1, axis=0)

    # Məlumatları xronoloji ardıcıllıqla çeşidləyirik
    return df.sort_index()
