#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:20:00 2020

@author: prageeth_wijewardhane
"""
import pandas as pd

df = pd.read_csv('bootstrapping_results_test/' + 'EGNN_Bootstrapping_average_results.csv', header='infer')
df['predicted labels'] = df['average'].round(0)
df.to_csv('bootstrapping_results_test/' + 'EGNN_Bootstrapping_average_results_with_predicted_labels.csv', header=True, index=None)