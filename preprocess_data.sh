#!/bin/bash

DATASET=PDL1
# DATASET=yourdata

radius=2
# radius=2

python preprocess_data_modifying.py $DATASET $radius
