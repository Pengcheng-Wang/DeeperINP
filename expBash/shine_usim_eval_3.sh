#!/usr/bin/env bash

## blstm for action pred in evaluation setting, Jan 20, 2018.
for rnnHdLc in 4
do
    echo 'in Bayesian LSTM hidden layer 4' ${rnnHdLc}
    date +%Y,%m,%d-%H:%M:%S
    for t in 1 2 3 4 5;
    do
        for alr in 1e-3;
        do
            g=0
            if [ $t -eq 2 ]
            then
                g=0
            fi
            if [ $t -eq 1 ];
            then
                th userSimMain.lua -trType ac -save blstm_alr_${alr}_hdlc_${rnnHdLc}_train_tr/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -learningRate ${alr} -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train_tr -uppModel rnn_blstm -lstmHist 10 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
            else
                th userSimMain.lua -trType ac -save blstm_alr_${alr}_hdlc_${rnnHdLc}_train_tr/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -learningRate ${alr} -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train_tr -uppModel rnn_blstm -lstmHist 10 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
            fi
        done
    done
    wait
    echo 'done with in Bayesian LSTM' ${rnnHdLc}
    date +%Y,%m,%d-%H:%M:%S
done