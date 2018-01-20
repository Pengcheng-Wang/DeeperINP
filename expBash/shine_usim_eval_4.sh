#!/usr/bin/env bash

### Action prediction using CNN models in evaluation mode. Jan 20, 2018.
s=1
for rnnHdLc in 1
do
    echo 'in CNN act prediction model round(layer)' ${rnnHdLc}
    date +%Y,%m,%d-%H:%M:%S
    for t in 1 2 3 4 5;
    do
        g=0
        if [ $t -eq 2 ]
        then
            g=0
        fi
        if [ $t -eq 1 ];
        then
            th userSimMain.lua -trType ac -save act8CnnV4_L${rnnHdLc}Lr5e-3_train_tr/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train_tr -uppModel cnn_uSimTempCnn -lstmHist 10 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 > /dev/null &
        else
            th userSimMain.lua -trType ac -save act8CnnV4_L${rnnHdLc}Lr5e-3_train_tr/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train_tr -uppModel cnn_uSimTempCnn -lstmHist 10 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 > /dev/null &
        fi
    done
    wait
    echo 'done with 4 sets in CNN act model' ${rnnHdLc}
    date +%Y,%m,%d-%H:%M:%S
done