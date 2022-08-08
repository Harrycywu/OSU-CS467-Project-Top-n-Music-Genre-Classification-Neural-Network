#!/bin/bash

i=1

for l in 0.001 0.0001 0.00001
do
    for b in 32 16 8 4
    do
        python main.py --id=$i --lr=$l --batch_size=$b --feature_type=mel_spectrogram
        let "i+=1"
    done
done
