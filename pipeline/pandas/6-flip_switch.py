#!/usr/bin/env python3
"""
Modul DataFrame-in tərs xronoloji sıralanması və transpozisiyası üçündür.
"""


def flip_switch(df):
    """
    Dataframe-i tərs xronoloji ardıcıllıqla sıralayır və transpozisiya edir.
    """
    return df.iloc[::-1].T
