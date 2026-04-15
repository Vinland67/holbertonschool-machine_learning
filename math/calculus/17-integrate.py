#!/usr/bin/env python3
"""Calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None

    # Giriş siyahısındakı hər bir elementin rəqəm olduğunu yoxla
    for x in poly:
        if not isinstance(x, (int, float)):
            return None

    # Yeni siyahı sənə verilən C rəqəmi ilə başlayır
    integral = [C]

    # İnteqral qaydası: əmsalı (indeks + 1)-ə bölürük
    for i in range(len(poly)):
        val = poly[i] / (i + 1)
        # Əgər nəticə tam rəqəmdirsə (məs: 5.0), onu integer (5) kimi saxla
        if val % 1 == 0:
            val = int(val)
        integral.append(val)

    # Siyahının sonundakı lazımsız sıfırları təmizləmək (as small as possible)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
