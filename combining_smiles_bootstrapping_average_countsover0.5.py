#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:33:25 2019

@author: prageeth_wijewardhane"""

import pandas as pd

data1 = pd.read_csv('bootstrapping_results/EGNN_Bootstrapping_average_results.csv')
data2 = pd.read_csv('bootstrapping_results/EGNN_Bootstrapping_counts_over_0.5.csv')

data = pd.concat([data1,data2], axis=1)
data.to_csv('bootstrapping_results/' + 'Final_bootstrapping_results_with_counts.csv', header=True, index=False)