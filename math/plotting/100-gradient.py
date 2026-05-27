#!/usr/bin/env python3
"""
Plots a 2D scatter plot of a mountain elevation with a colorbar
"""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    Renders the mountain elevation data points with intensity mapping
    """
    np.random.seed(5)
    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    points = plt.scatter(x, y, c=z, cmap='viridis')

    cbar = plt.colorbar(points)
    cbar.set_label('elevation (m)')

    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')

    plt.show()
