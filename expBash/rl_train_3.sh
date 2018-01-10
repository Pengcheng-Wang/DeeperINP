#!/usr/bin/env bash

## Train A3C RL models using Ep50 usp with sampling temperature of 0.33 0.2 0.45
echo 'A3C RL training test'
date +%Y,%m,%d-%H:%M:%S
hist=3
for thrN in 6;
do
    for lrt in 7e-4;
    do
        for sfxTem in 0.33 0.2 0.45;
        do
            th rlMain.lua -async A3C -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 64 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*1000000)) -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23Ep50.t7 -_id ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem} -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
            wait
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
        done
    done
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S