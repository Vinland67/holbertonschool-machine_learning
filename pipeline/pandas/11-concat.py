#!/usr/bin/env python3
"""
Modul iki DataFrame-i indeksləyib birləşdirmək üçün funksiya təmin edir.
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Dataframe-ləri Timestamp üzrə indeksləyir və şərtə uyğun concat edir.
    """
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # 1417411920 daxil olmaqla df2-ni filtrləyirik
    df2_filtered = df2_indexed.loc[:1417411920]

    # Sətir uzunluğunu qorumaq üçün pd.concat-ı bölürük
    return pd.concat(
        [df2_filtered, df1_indexed],
        keys=['bitstamp', 'coinbase']
    )
