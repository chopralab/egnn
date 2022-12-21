#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:35:57 2020

@author: prageeth_wijewardhane
"""
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, cohen_kappa_score

df = pd.read_csv("test/radius2/correct_properties_no_tensor.csv", header = None)
l =df.values.tolist()
correct_labels = sum(l, [])
print(l)

def f1_score(precision, recall):
    if precision==0 and recall == 0:
        F1_score = 'undefined'
    else:
        F1_score = 2 * ((precision * recall) / (precision + recall))
    return F1_score



list_AUC = []
list_precision = []
list_recall = []
list_f1 = []
list_kappa = []

for file in os.listdir("bootstrapping_results_test/"):
    results = pd.read_csv("bootstrapping_results_test/"+ file, header = 0)
    print(results)

    l2 = results.values.tolist()
#    print(l2)

    predicted_scores = []
    for l in l2:
        predicted_scores.append(l[2])
    
    predicted_labels = []
    for l in l2:
        predicted_labels.append(l[1])
    
    print(len(correct_labels), len(predicted_scores), len(predicted_labels))

    AUC = roc_auc_score(correct_labels, predicted_scores)
    precision = precision_score(correct_labels, predicted_labels)
    recall = recall_score(correct_labels, predicted_labels)
    F1_score = f1_score(precision,recall)
    kappa = cohen_kappa_score(correct_labels, predicted_labels)
    
#    data= [AUC, precision, recall, F1_score, kappa]
    
    list_AUC.append(AUC)
    list_precision.append(precision)
    list_recall.append(recall)
    list_f1.append(F1_score)
    list_kappa.append(kappa)
    
    
    print("AUC =",AUC , "precision =",precision, "recall =",recall, "F1_score =",F1_score, "kappa =",kappa)
    
d = {"AUC": list_AUC , "precision": list_precision, "recall": list_recall, "F1_score": list_f1, "kappa": list_kappa}
print('dictionary', d)
df2 = pd.DataFrame(d)
print("dataframe2:",df2)
#df2.to_csv("all_bootstrapped_models_performances.csv",  header=True, index=False)

print('mean:', df2.mean(axis = 0, skipna = True))
print('std:', df2.std(axis = 0, skipna = True))
