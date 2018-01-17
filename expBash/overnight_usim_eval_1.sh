#!/usr/bin/env bash


# Re-train cnnmoe for score prediction.
# Try to do score/outcome prediction using cnn moe models.
# This set performs well. Especially for cnnmoe_L2H2K1E30_exp_23_sh. Should pay attention to this.
s=1
for moeExpCnt in 23
do
    for recD in 5
    do
        echo 'For outcome prediction CNN-moe round' ${moeExpCnt}
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
                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K1E30_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd_retrain/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
            else
                th userSimMain.lua -trType sc -save sc_cnnmoe_L2H2K1E30_exp_${moeExpCnt}_sh/rnndrop.1/seed$(($s))/augRnd_retrain/tdiv$(($t))/ -batchSize 160 -coefL2 1e-2 -rnnHdSizeL1 21 -rnnHdLyCnt 2 -moeExpCnt $(($moeExpCnt)) -ciuTType train -uppModel cnn_uSimCnn_moe -uSimScSoft 1 -lstmHist 2 -cnnKernelWidth 1 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -learningRate 5e-5 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) -cnnConnType v4 -scorePredStateScope 30 > /dev/null &
            fi
        done
        wait
        echo 'done with 5 sets in CNN-moe L2'  ${moeExpCnt}
        date +%Y,%m,%d-%H:%M:%S
    done
done



## Evaluating act prediction models,
echo 'Bayesian RHN Evaluation with data augmentation (Retrain)'
date +%Y,%m,%d-%H:%M:%S
for rnnHdLc in 3;
do
    for rhnRD in 3;
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
                    th userSimMain.lua -trType ac -save ac_rhn_alr_${alr}_hdlc_${rnnHdLc}_rhnRD_${rhnRD}_noDataAug/rnndrop.1/seed$(($s))/augRnd_retrain/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -learningRate ${alr} -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept ${rhnRD} -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 2000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
                else
                    th userSimMain.lua -trType ac -save ac_rhn_alr_${alr}_hdlc_${rnnHdLc}_rhnRD_${rhnRD}_noDataAug/rnndrop.1/seed$(($s))/augRnd_retrain/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -learningRate ${alr} -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept ${rhnRD} -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 2000  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
                fi
            done
        done
    done
done
wait
echo 'Done with in Bayesian RHN no data augmentation'
date +%Y,%m,%d-%H:%M:%S