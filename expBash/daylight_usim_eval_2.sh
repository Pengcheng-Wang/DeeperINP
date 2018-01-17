#!/usr/bin/env bash

## Evaluating act prediction models,
echo 'Bayesian RHN Evaluation with data augmentation (Retrain)'
date +%Y,%m,%d-%H:%M:%S
s=0
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
                    th userSimMain.lua -trType ac -save ac_rhn_alr_${alr}_hdlc_${rnnHdLc}_rhnRD_${rhnRD}_retrain/rnndrop.1/seed$(($s))/augRnd_retrain/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -learningRate ${alr} -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept ${rhnRD} -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
                else
                    th userSimMain.lua -trType ac -save ac_rhn_alr_${alr}_hdlc_${rnnHdLc}_rhnRD_${rhnRD}_retrain/rnndrop.1/seed$(($s))/augRnd_retrain/tdiv$(($t))/ -batchSize 160 -coefL2 5e-3 -learningRate ${alr} -rnnHdSizeL1 21 -rnnHdLyCnt $(($rnnHdLc)) -rhnReccDept ${rhnRD} -ciuTType train -uppModel rnn_rhn -lstmHist 10 -usimTrIte 1500  -uSimShLayer 0 -testSetDivSeed $(($t-1)) -gpu $(($g)) -dropoutUSim 0.1 -uSimLstmBackLen 3 -actPredDataAug 1 -seed $(($s)) > /dev/null &
                fi
            done
        done
    done
done
wait
echo 'Done with in Bayesian RHN no data augmentation'
date +%Y,%m,%d-%H:%M:%S