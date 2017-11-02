#!/usr/bin/env bash

### Train action prediction models
#cd ../
### The following experiment compare the effects of data augmentation without batch normalization and dropout
#for s in `seq 1 10`;
#do
#    ## Train models without data augmentation
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ];
#        then
#            g=1
#        fi
#        th userSimMain.lua -trType ac -save seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -actPredDataAug 0 -seed $(($s)) &
#    done
#    wait
#    ## Train models with data augmentation
#    for t in `seq 1 5`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=1
#        fi
#        th userSimMain.lua -trType ac -save seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -actPredDataAug 1 -seed $(($s)) &
#    done
#    wait
#done

### The following experiment compare the effects of data augmentation without batch normalization but with dropout
for s in `seq 1 2`;
do
    ## Train models without data augmentation
    for t in `seq 1 5`;
    do
        g=0
        if [ $t -eq 2 ];
        then
            g=1
        fi
        th userSimMain.lua -trType ac -save rnndrop/seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 800  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.25 -actPredDataAug 0 -seed $(($s)) &
    done
    wait
    ## Train models with data augmentation
    for t in `seq 1 5`;
    do
        g=0
        if [ $t -eq 2 ]
        then
            g=1
        fi
        th userSimMain.lua -trType ac -save rnndrop/seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 800  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.25 -actPredDataAug 1 -seed $(($s)) &
    done
    wait
done