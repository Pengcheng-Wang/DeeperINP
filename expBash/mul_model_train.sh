#!/usr/bin/env bash

#### Dec 8, 2017, Try the CNN models for outcome prediction
#s=1
#for rnnHdLc in 1 2 3
#do
#    echo 'in CNN outcome prediction model round(layer)' ${rnnHdLc}
#    date +%Y,%m,%d-%H:%M:%S
#    for t in `seq 1`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=0
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType sc -save sc8CnnV1_L${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel cnn_uSimTempCnn -lstmHist 10 -usimTrIte 2500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v1 > /dev/null &
#        else
#            th userSimMain.lua -trType sc -save sc8CnnV1_L${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel cnn_uSimTempCnn -lstmHist 10 -usimTrIte 2500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v1 > /dev/null &
#        fi
#    done
#    for t in `seq 1`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=0
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType sc -save sc8CnnV2_L${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel cnn_uSimTempCnn -lstmHist 10 -usimTrIte 2500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v2 > /dev/null &
#        else
#            th userSimMain.lua -trType sc -save sc8CnnV2_L${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel cnn_uSimTempCnn -lstmHist 10 -usimTrIte 2500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v2 > /dev/null &
#        fi
#    done
#    for t in `seq 1`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=0
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType sc -save sc8CnnV3_L${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel cnn_uSimTempCnn -lstmHist 10 -usimTrIte 2500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v3 > /dev/null &
#        else
#            th userSimMain.lua -trType sc -save sc8CnnV3_L${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel cnn_uSimTempCnn -lstmHist 10 -usimTrIte 2500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v3 > /dev/null &
#        fi
#    done
#    for t in `seq 1`;
#    do
#        g=0
#        if [ $t -eq 2 ]
#        then
#            g=0
#        fi
#        if [ $t -eq 1 ];
#        then
#            th userSimMain.lua -trType sc -save sc8CnnV4_L${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel cnn_uSimTempCnn -lstmHist 10 -usimTrIte 2500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 > /dev/null &
#        else
#            th userSimMain.lua -trType sc -save sc8CnnV4_L${rnnHdLc}/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -ciuTType train -uppModel cnn_uSimTempCnn -lstmHist 10 -usimTrIte 2500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.15 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 > /dev/null &
#        fi
#    done
#    wait
#    echo 'done with 4 sets in CNN outcome model' ${rnnHdLc}
#    date +%Y,%m,%d-%H:%M:%S
#done

## Dec 11, 2017.
## Try to do score/outcome prediction using rhn models, without using multi-task structure. This time with shorter valid score prediction sequence
#s=1
#for rnnHdLc in 1 2 3
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in B-LSTM round' ${rnnHdLc}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in `seq 1 2`;
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=0
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_blstm_hdlc_${rnnHdLc}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_blstm -lstmHist 10 -usimTrIte 5000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 1e-4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -scorePredStateScope 15 &
#            else
#                th userSimMain.lua -trType sc -save sc_blstm_hdlc_${rnnHdLc}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept $(($recD)) -ciuTType train -uppModel rnn_blstm -lstmHist 10 -usimTrIte 5000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 1e-4 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -scorePredStateScope 15 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in B-LSTM'  ${rnnHdLc}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done


## Dec 12, 2017.
## Try to do score/outcome prediction using cnn moe models. Looks like kernel width 1 does not help.
#s=1
#for moeExpCnt in 16 24 32
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in 1 2 3 4 5
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2K1_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 3 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 &
#            else
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2K1_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 3 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in CNN-moe L2'  ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done


## Dec 12, 2017.
## Try to do score/outcome prediction using cnn moe models. Try use kernel width 2 with 5 frames, 2-layer cnn
#s=1
#for moeExpCnt in 16 24 32
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in 1 2 3 4 5
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=1
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2K1_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 5 -cnnKernelWidth 2 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 &
#            else
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2K1_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 5 -cnnKernelWidth 2 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in CNN-moe L2'  ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done

## Dec 14, 2017.
## Try to do score/outcome prediction using cnn moe models.
## Try L2H2K2 with larger hist scope
#s=1
#for moeExpCnt in 4 8 16 24 32
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in 1 2 3 4 5
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=0
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K2E60_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 2 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 60 &
#            else
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K2E60_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 2 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 60 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in CNN-moe L2H2K2E30'  ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done

#
## Dec 15, 2017.
## Try to do score/outcome prediction using cnn moe models.
## Try L2H2K1 with smaller hist scope, cnn structure v2.
#s=1
#for moeExpCnt in 8 16 24 32
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in 1 2 3 4 5
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=0
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K1E20_cv2_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v2 -scorePredStateScope 20 &
#            else
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K1E20_cv2_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v2 -scorePredStateScope 20 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in CNN-moe L2H2K1E20_cv2'  ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done


## Dec 15, 2017.
## Try to do score/outcome prediction using cnn moe models.
## Try L2H2K1 with even smaller hist scope of 10, cnn structure v2.
#s=1
#for moeExpCnt in 8 16 24 32 48
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in 1 2 3 4 5
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=0
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K1E10_cv2_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v2 -scorePredStateScope 10 &
#            else
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K1E10_cv2_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v2 -scorePredStateScope 10 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in CNN-moe L2H2K1E10_cv2'  ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done


## Dec 16, 2017.
## Try to do score/outcome prediction using cnn moe models.
## Try L2H2K1 with E30, and with different cnn structure, like v2.
#s=1
#for moeExpCnt in 8 16 23 24 32
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in 1 2 3 4 5
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=0
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K1E30_cv2_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v2 -scorePredStateScope 30 &
#            else
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K1E30_cv2_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v2 -scorePredStateScope 30 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in CNN-moe L2H2K1E10_cv2'  ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done


## Dec 16, 2017.
## Try to do score/outcome prediction using cnn moe models.
## Try L2H2K1 with E30, and with different cnn structure, like v1.
#s=1
#for moeExpCnt in 8 16 23 24 32
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in 1 2 3 4 5
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=0
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K1E30_cv1_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v1 -scorePredStateScope 30 &
#            else
#                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K1E30_cv1_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v1 -scorePredStateScope 30 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in CNN-moe L2H2K1E10_cv1'  ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done


## Dec 17, 2017.
## Try to do score/outcome prediction using cnn moe models.
## Try to utilize various error weights for the score regressor. Try 0.25 here
#s=1
#for moeExpCnt in 8 16 23 24 32
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in 1 2 3 4 5
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=0
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_cnnmoe_wt025_L2H2K1E30_cv4_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 0.25 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 &
#            else
#                th userSimMain.lua -trType sc -save sc_cnnmoe_wt025_L2H2K1E30_cv4_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 0.25 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in CNN-moe L2H2K1E10_cv1'  ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done
#wait
## Dec 17, 2017.
## Try to do score/outcome prediction using cnn moe models.
## Try to utilize various error weights for the score regressor. Try 0.1 here
#s=1
#for moeExpCnt in 8 16 23 24 32
#do
#    for recD in 5
#    do
#        echo 'For outcome prediction, in CNN-moe round' ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#        for t in 1 2 3 4 5
#        do
#            g=0
#            if [ $t -eq 2 ]
#            then
#                g=0
#            fi
#            if [ $t -eq 1 ];
#            then
#                th userSimMain.lua -trType sc -save sc_cnnmoe_wt010_L2H2K1E30_cv4_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 0.1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 &
#            else
#                th userSimMain.lua -trType sc -save sc_cnnmoe_wt010_L2H2K1E30_cv4_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 0.1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
#            fi
#        done
#        wait
#        echo 'done with 2 sets in CNN-moe L2H2K1E10_cv1'  ${moeExpCnt}
#        date +%Y,%m,%d-%H:%M:%S
#    done
#done


# Dec 18, 2017.
# Try to do score/outcome prediction using cnn moe models.
# Try to utilize various error weights for the score regressor. Try 0.25 here
# Also try to do Bayesian sampling in evaluation
s=1
for moeExpCnt in 8 16 23 24 32
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
                th userSimMain.lua -trType sc -save sc_cnnmoe_wt025_L2H2K1E30_cv4_exp_${moeExpCnt}_smp100_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 0.25 -uSimBayesEvl 100 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 &
            else
                th userSimMain.lua -trType sc -save sc_cnnmoe_wt025_L2H2K1E30_cv4_exp_${moeExpCnt}_smp100_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 0.25 -uSimBayesEvl 100 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
            fi
        done
        wait
        echo 'done with 2 sets in CNN-moe L2H2K1E10_cv1'  ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
    done
done
wait
# Dec 17, 2017.
# Try to do score/outcome prediction using cnn moe models.
# Try to utilize various error weights for the score regressor. Try 0.1 here
s=1
for moeExpCnt in 8 16 23 24 32
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
                th userSimMain.lua -trType sc -save sc_cnnmoe_wt010_L2H2K1E30_cv4_exp_${moeExpCnt}_smp100_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 0.1 -uSimBayesEvl 100 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 &
            else
                th userSimMain.lua -trType sc -save sc_cnnmoe_wt010_L2H2K1E30_cv4_exp_${moeExpCnt}_smp100_sh/rnndrop.1/seed$(($s))/augRnd/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 0.1 -uSimBayesEvl 100 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
            fi
        done
        wait
        echo 'done with 2 sets in CNN-moe L2H2K1E10_cv1'  ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
    done
done