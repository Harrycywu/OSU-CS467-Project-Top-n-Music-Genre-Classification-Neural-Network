#!/bin/bash

i=1

for f in 'wavelet' 'spectrogram' 'mel_spectrogram' 'mfcc'
do
    python ../train.py --id=$i --feature_type=$f --lr=0.00001
    let "i+=1"
done
