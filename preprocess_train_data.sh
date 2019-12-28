#!/bin/bash

DATASET=PDL1
# DATASET=yourdata

radius=2
# radius=2

python preprocess_train_data.py $DATASET $radius
