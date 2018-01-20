#!/usr/bin/env bash

## Grid-LSTM for action prediction. 4 hidden layer, learning rate 2e-3. Jan 16, 2018.
echo 'in GridLSTM evaluating'
date +%Y,%m,%d-%H:%M:%S
for rnnHdLc in 4;
do
    for t in 1 2 3 4 5;
    do
        for alr in 2e-3;
        do
            g=0
            if [ $t -eq 2 ]
            then
                g=0
            fi
            if [ $t -eq 1 ];
            then
                th userSimMain.lua -trType ac -save ac_gridlstm_alr_${alr}_hdlc_${rnnHdLc}_train_tr/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -learningRate ${alr} -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train_tr -uppModel rnn_bGridlstm -lstmHist 10 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
            else
                th userSimMain.lua -trType ac -save ac_gridlstm_alr_${alr}_hdlc_${rnnHdLc}_train_tr/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -learningRate ${alr} -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train_tr -uppModel rnn_bGridlstm -lstmHist 10 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
            fi
        done
    done
done
wait
echo 'Done with GridLSTM evaluating'
date +%Y,%m,%d-%H:%M:%S