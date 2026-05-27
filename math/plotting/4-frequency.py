#!/usr/bin/env python3
"""
Defines frequency function
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades with bins every 10 units
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # 0-dan 100-ə qədər hər 10 vahiddən bir bin aralıqları yaradırıq
    bins = np.arange(0, 101, 10)

    # Histogram çəkirik, edgecolor='black' kənarları qara edir
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Oxların diapazonlarını tam şəkildəki kimi məhdudlaşdırırıq
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    # Oxların adlarını, başlıqları və alt hissədəki x oxu bölgülərini qoyuruq
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xticks(bins)

    # Qrafiki göstəririk
    plt.show()
