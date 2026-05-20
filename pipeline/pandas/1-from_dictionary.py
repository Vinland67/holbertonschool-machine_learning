#!/usr/bin/env python3
"""
Bu modul verilmiş lüğətdən (dictionary) Pandas DataFrame yaradır.
"""
import pandas as pd

data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

row_labels = ['A', 'B', 'C', 'D']

df = pd.DataFrame(data, index=row_labels)
