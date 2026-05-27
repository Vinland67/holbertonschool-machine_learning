#!/usr/bin/env python3
"""
3D Visualization of the Iris Dataset using Principal Component Analysis
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot_pca():
    """
    Performs SVD and plots the 3D projection of Iris dataset
    """
    lib = np.load("pca.npz")
    data = lib["data"]
    labels = lib["labels"]

    data_means = np.mean(data, axis=0)
    norm_data = data - data_means
    _, _, Vh = np.linalg.svd(norm_data)
    pca_data = np.matmul(norm_data, Vh[:3].T)

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111, projection='3d')

    x_vals = pca_data[:, 0]
    y_vals = pca_data[:, 1]
    z_vals = pca_data[:, 2]

    ax.scatter(x_vals, y_vals, z_vals, c=labels, cmap='plasma')

    ax.set_xlabel('U1')
    ax.set_ylabel('U2')
    ax.set_zlabel('U3')
    
    ax.set_title('PCA of Iris Dataset')

    plt.show()


if __name__ == "__main__":
    plot_pca()
