#!/usr/bin/env bash

# Dec 11, 2017.
# Try to do score/outcome prediction using cnn moe models
s=1
for moeExpCnt in 16 24 32
do
    for recD in 5
    do
        echo 'For outcome prediction, in B-LSTM round' ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
        for t in `seq 1 2 3`;
        do
            g=0
            if [ $t -eq 2 ]
            then
                g=0
            fi
            if [ $t -eq 1 ];
            then
                th userSimMain.lua -trType sc -save sc_cnnmoe_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 1 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 0 -lstmHist 3 -usimTrIte 5000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 1e-4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 15 &
            else
                th userSimMain.lua -trType sc -save sc_cnnmoe_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 1 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 0 -lstmHist 3 -usimTrIte 5000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 1e-4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 15 > /dev/null &
            fi
        done
        wait
        echo 'done with 2 sets in B-LSTM'  ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
    done
done