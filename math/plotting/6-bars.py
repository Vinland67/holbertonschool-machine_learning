#!/usr/bin/env python3
"""
Generates a stacked bar chart for fruit distribution per person
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots stacked bars representing quantities of various fruits
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruit_types = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    w = 0.5
    indices = np.arange(len(people))
    current_bottom = np.zeros(len(people))

    for idx in range(len(fruit_types)):
        plt.bar(indices, fruit[idx], width=w, bottom=current_bottom,
                color=colors[idx], label=fruit_types[idx])
        current_bottom += fruit[idx]

    plt.xticks(indices, people)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend(loc='upper right')

    plt.show()
