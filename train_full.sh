#!/bin/bash
#SBATCH -A gchopra
#SBATCH --nodes=1
#SBATCH -n 20 
#SBATCH --time=48:00:00


DATASET=PDL1
# DATASET=yourdata

radius=2
# radius=2

update=sum
# update=mean

# output=sum
output=mean

dim=10
hidden_layer=1
output_layer=2
batch=15
lr=1e-3
lr_decay=0.9
decay_interval=10
weight_decay=1e-6
iteration=100

setting=$DATASET--radius$radius--update_$update--output_$output--dim$dim--hidden_layer$hidden_layer--output_layer$output_layer--batch$batch--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--weight_decay$weight_decay--iteration$iteration
python train_full_bootstrapping.py $DATASET $radius $update $output $dim $hidden_layer $output_layer $batch $lr $lr_decay $decay_interval $weight_decay $iteration $setting
