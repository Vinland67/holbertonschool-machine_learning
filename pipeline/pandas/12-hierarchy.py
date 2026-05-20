#!/usr/bin/env python3
"""
Modul MultiIndex qurmaq üçün funksiya təmin edir.
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Dataframe-ləri birləşdirir və iyerarxiya qurur.
    """
    df1_idx = index(df1)
    df2_idx = index(df2)

    # Zaman aralığını filtrləyirik
    st, en = 1417411980, 1417417980
    df1_f = df1_idx.loc[st:en]
    df2_f = df2_idx.loc[st:en]

    # Qısa sətirlərlə concat edirik
    df = pd.concat(
        [df2_f, df1_f],
        keys=['bitstamp', 'coinbase']
    )

    # İndeks səviyyələrinin yerini dəyişirik
    df = df.swaplevel(0, 1, axis=0)

    # Xronoloji ardıcıllıqla çeşidləyirik
    return df.sort_index()
