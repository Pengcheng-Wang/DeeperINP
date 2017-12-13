#!/usr/bin/env bash

## Dec 11, 2017.
## Try to do score/outcome prediction using simple moe models
#s=1
#for moeExpCnt in 16 24 32
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in B-LSTM round' ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in `seq 1 2 3`;
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=0
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_moe_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel moe -lstmHist 10 -usimTrIte 1000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 1e-4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -scorePredStateScope 15 &
#            else
#                th userSimMain.lua -trType sc -save sc_moe_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel moe -lstmHist 10 -usimTrIte 1000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 1e-4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -scorePredStateScope 15 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in B-LSTM'  ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done

## Dec 11, 2017.
## Try to do score/outcome prediction using cnn moe models
#s=1
#for moeExpCnt in 16 24 32
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in 3 4 5
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 3 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 &
#            else
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 3 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in CNN-moe L2'  ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done

## Dec 12, 2017. Night
## The 2-layer model performs not bad. And it seems cnn kernel width 3 is better than kernel width 1 (when input frame is 3). It looks like 2-layer cnn may be better than 1-layer? Not sure. Try 2 layer with larger hist length.
## This does not perform well. It looks like too long history embedded is not helpful.
#s=1
#for moeExpCnt in 16 24 32
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in 1 2 3
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H5K3_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 5 -cnnKernelWidth 3 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 &
#            else
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H5K3_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 5 -cnnKernelWidth 3 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in CNN-moe L2'  ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done

# Dec 13, 2017.
# The 2-layer model performs not bad. And it seems cnn kernel width 3 is better than kernel width 1 (when input frame is 3). It looks like 2-layer cnn may be better than 1-layer? Not sure. Try 2 layer with larger hist length.
# This does not perform well. It looks like too long history embedded is not helpful.
s=1
for moeExpCnt in 8 16 24 32
do
    for recD in 5
    do
        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
        for t in 1 2 3
        do
            g=0
            if [ $t -eq 2 ]
            then
                g=0
            fi
            if [ $t -eq 1 ];
            then
                th userSimMain.lua -trType sc -save sc_cnnmoe_L3H2K1E30_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 &
            else
                th userSimMain.lua -trType sc -save sc_cnnmoe_L3H2K1E30_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
            fi
        done
        wait
        echo 'done with 3 sets in CNN-moe L3'  ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
    done
done