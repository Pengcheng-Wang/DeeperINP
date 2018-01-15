#!/usr/bin/env bash

## Rerun these experiments to get full 5-fold results
# Dec 11, 2017.
# Try to do score/outcome prediction using rhn models, without using multi-task structure.
s=1
for rnnHdLc in 1
do
    for recD in 5
    do
        echo 'For outcome prediction, in RNN round' ${recD}
        date +%Y,%m,%d-%H:%M:%S
        for t in `seq 3 4 5`;
        do
            g=0
            if [ $t -eq 2 ]
            then
                g=1
            fi
            if [ $t -eq 1 ];
            then
                th userSimMain.lua -trType sc -save sc_rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 1e-4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) &
            else
                th userSimMain.lua -trType sc -save sc_rhn_hdlc_${rnnHdLc}_recDp_${recD}L/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 1e-4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
            fi
        done
#        wait
        echo 'done with 2 sets in RHN'  ${recD}
        date +%Y,%m,%d-%H:%M:%S
    done
done


# Dec 12, 2017.
# Try to do score/outcome prediction using cnn moe models.
# This set of exp performs poorly.
s=1
for moeExpCnt in 4
do
    for recD in 5
    do
        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
        for t in 4 5
        do
            g=0
            if [ $t -eq 2 ]
            then
                g=0
            fi
            if [ $t -eq 1 ];
            then
                th userSimMain.lua -trType sc -save sc_rhnmoe_L1_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 1 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel rnn_rhn_moe -uSimScSoft 1 -lstmHist 10 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 2e-4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 &
            else
                th userSimMain.lua -trType sc -save sc_rhnmoe_L1_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 1 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel rnn_rhn_moe -uSimScSoft 1 -lstmHist 10 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 2e-4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
            fi
        done
        wait
        echo 'done with 2 sets in CNN-moe L2'  ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
    done
done