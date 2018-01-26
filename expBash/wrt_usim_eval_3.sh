#!/usr/bin/env bash

# Train cnn model for sc without either moe or soft-score
# Try to do score/outcome prediction using cnn moe models.
# This set performs well. Especially for cnnmoe_L2H2K1E30_exp_23_sh. Should pay attention to this.
s=1
for moeExpCnt in 23
do
    for recD in 5
    do
        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
        for t in 1 2 3 4 5
        do
            g=0
            if [ $t -eq 2 ]
            then
                g=0
            fi
            if [ $t -eq 1 ];
            then
                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K1E30_exp_${moeExpCnt}_sh_noSoft/rnndrop.1/seed$(($s))/augRnd_re/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 0 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 &
            else
                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K1E30_exp_${moeExpCnt}_sh_noSoft/rnndrop.1/seed$(($s))/augRnd_re/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 0 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
            fi
        done
        wait
        echo 'done with 5 sets in CNN-moe L2'  ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
    done
done