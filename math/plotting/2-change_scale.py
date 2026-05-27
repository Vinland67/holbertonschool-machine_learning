#!/usr/bin/env python3
"""
Defines change_scale function
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots the exponential decay of C-14 with a logarithmic y-axis
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Qrafiki çəkirik
    plt.plot(x, y)

    # Y oxunu logaritmik miqyasa keçiririk
    plt.yscale('log')

    # X oxunun sərhədlərini tam təyin edirik
    plt.xlim(0, 28650)

    # Oxların adlarını və başlığı qoyuruq
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')

    # Qrafiki göstəririk
    plt.show()
