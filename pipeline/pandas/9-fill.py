#!/usr/bin/env python3
"""
Modul DataFrame daxilindəki boş (NaN) dəyərləri doldurmaq üçün funksiya təmin edir.
"""


def fill(df):
    """
    NaN dəyərləri şərtlərə uyğun doldurur və Weighted_Price sütununu silir.
    """
    # 1. Weighted_Price sütununu silirik
    df = df.drop(columns=['Weighted_Price'])

    # 2. Close sütunundakı NaN dəyərləri əvvəlki sətrin dəyəri ilə doldururuq
    df['Close'] = df['Close'].ffill()

    # 3. High, Low və Open sütunlarını eyni sətirdəki Close dəyəri ilə doldururuq
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # 4. Həcm sütunlarındakı NaN dəyərləri 0 ilə əvəzləyirik
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
