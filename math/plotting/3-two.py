#!/usr/bin/env python3
"""
Defines two function
"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plots exponential decay of two radioactive elements with a legend
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Xətləri çəkirik və legend üçün label təyin edirik
    plt.plot(x, y1, 'r--', label='C-14')
    plt.plot(x, y2, 'g-', label='Ra-226')

    # Oxların sərhədlərini təyin edirik
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Oxların adlarını və başlığı qoyuruq
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')

    # Legend-i sağ yuxarı küncə yerləşdiririk
    plt.legend(loc='upper right')

    # Qrafiki göstəririk
    plt.show()
