#!/bin/bash

i=1

for l in 0.001 0.0001 0.00001
do
    for b in 32 16 8 4
    do
        python ../train.py --id=$i --lr=$l --batch_size=$b
        let "i+=1"
    done
done
