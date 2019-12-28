#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:19:05 2019

@author: prageeth_wijewardhane
"""

import pandas as pd

def count_values_in_range(series, range_min, range_max):
    return series.between(left=range_min, right=range_max).sum()


data = pd.read_csv('bootstrapping_results/all_bootstrapping_predictions.csv')
print(data)

data['number_of_actives_higher_than_0.5'] = data.apply(func=lambda row: count_values_in_range(row, 0.5, 1.0), axis=1)

data['number_of_actives_higher_than_0.5'].to_csv('bootstrapping_results/' + 'EGNN_Bootstrapping_counts_over_0.5.csv', header=True, index=False)