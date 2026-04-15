#!/usr/bin/env python3
"""Calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    
    # Əgər poly-nin daxilindəkilər rəqəm deyilsə None qaytar
    for x in poly:
        if not isinstance(x, (int, float)):
            return None

    # Əgər çoxhədli sabit rəqəmdirsə (məs: f(x) = 5), törəməsi [0]-dır
    if len(poly) == 1:
        return [0]

    derivative = []
    # Törəmə: hər bir əmsalı öz qüvvətinə (indeksinə) vururuq
    # Sabit rəqəmin törəməsi 0 olduğu üçün dövrü 1-ci indeksdən başlayırıq
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])

    # Siyahının sonundakı lazımsız sıfırları təmizləmək (trailing zeros)
    # Amma əgər siyahı tamamilə boş qalsa [0] qaytar
    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    return derivative
