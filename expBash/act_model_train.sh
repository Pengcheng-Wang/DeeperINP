#!/usr/bin/env bash

# Train action prediction models
cd ../
for s in `seq 1 10`;
do
    ## Train models without data augmentation
    for t in `seq 1 5`;
    do
        g=0
        if [ $t -eq 2 ];
        then
            g=1
        fi
        th userSimMain.lua -trType ac -save seed$(($s))/no_aug/tdiv$(($t))/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -actPredDataAug 0 -seed $(($s)) &
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
        th userSimMain.lua -trType ac -save seed$(($s))/aug/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0 -actPredDataAug 1 -seed $(($s)) &
    done
    wait
done
## Seed is 1
## Train models without data augmentation
#th userSimMain.lua -trType ac -save seed1/no_aug/tem1/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 0 -gpu 0 -dropoutUSim 0 -actPredDataAug 0 -seed 1 &
#th userSimMain.lua -trType ac -save seed1/no_aug/tem2/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 1 -gpu 1 -dropoutUSim 0 -actPredDataAug 0 -seed 1 &
#th userSimMain.lua -trType ac -save seed1/no_aug/tem3/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 2 -gpu 0 -dropoutUSim 0 -actPredDataAug 0 -seed 1 &
#th userSimMain.lua -trType ac -save seed1/no_aug/tem4/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 3 -gpu 0 -dropoutUSim 0 -actPredDataAug 0 -seed 1 &
#th userSimMain.lua -trType ac -save seed1/no_aug/tem5/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 4 -gpu 0 -dropoutUSim 0 -actPredDataAug 0 -seed 1 &
#wait
## Train models with data augmentation
#th userSimMain.lua -trType ac -save seed1/aug/tem1/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 0 -gpu 0 -dropoutUSim 0 -actPredDataAug 1 -seed 1 &
#th userSimMain.lua -trType ac -save seed1/aug/tem2/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 1 -gpu 1 -dropoutUSim 0 -actPredDataAug 1 -seed 1 &
#th userSimMain.lua -trType ac -save seed1/aug/tem3/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 2 -gpu 0 -dropoutUSim 0 -actPredDataAug 1 -seed 1 &
#th userSimMain.lua -trType ac -save seed1/aug/tem4/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 3 -gpu 0 -dropoutUSim 0 -actPredDataAug 1 -seed 1 &
#th userSimMain.lua -trType ac -save seed1/aug/tem5/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 4 -gpu 0 -dropoutUSim 0 -actPredDataAug 1 -seed 1 &
## Seed is 2
## Train models without data augmentation
#th userSimMain.lua -trType ac -save seed2/no_aug/tem1/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 0 -gpu 0 -dropoutUSim 0 -actPredDataAug 0 -seed 1 &
#th userSimMain.lua -trType ac -save seed2/no_aug/tem2/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 1 -gpu 1 -dropoutUSim 0 -actPredDataAug 0 -seed 1 &
#th userSimMain.lua -trType ac -save seed2/no_aug/tem3/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 2 -gpu 0 -dropoutUSim 0 -actPredDataAug 0 -seed 1 &
#th userSimMain.lua -trType ac -save seed2/no_aug/tem4/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 3 -gpu 0 -dropoutUSim 0 -actPredDataAug 0 -seed 1 &
#th userSimMain.lua -trType ac -save seed2/no_aug/tem5/ -batchSize 80 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 4 -gpu 0 -dropoutUSim 0 -actPredDataAug 0 -seed 1 &
#wait
## Train models with data augmentation
#th userSimMain.lua -trType ac -save seed2/aug/tem1/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 0 -gpu 0 -dropoutUSim 0 -actPredDataAug 1 -seed 1 &
#th userSimMain.lua -trType ac -save seed2/aug/tem2/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 1 -gpu 1 -dropoutUSim 0 -actPredDataAug 1 -seed 1 &
#th userSimMain.lua -trType ac -save seed2/aug/tem3/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 2 -gpu 0 -dropoutUSim 0 -actPredDataAug 1 -seed 1 &
#th userSimMain.lua -trType ac -save seed2/aug/tem4/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 3 -gpu 0 -dropoutUSim 0 -actPredDataAug 1 -seed 1 &
#th userSimMain.lua -trType ac -save seed2/aug/tem5/ -batchSize 160 -coefL2 5e-3 -lstmHd 64 -ciuTType train -uppModel lstm -lstmHist 10 -usimTrIte 400  -uSimShLayer 0 -testSetDivSeed 4 -gpu 0 -dropoutUSim 0 -actPredDataAug 1 -seed 1 &