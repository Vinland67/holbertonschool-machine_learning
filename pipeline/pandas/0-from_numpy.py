#!/usr/bin/env python3
"""
Creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Converts a numpy array into a pandas dataframe
    with alphabetical column labels.
    """
    num_cols = array.shape[1]
    columns = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=columns)
