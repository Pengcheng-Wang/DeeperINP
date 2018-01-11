#!/usr/bin/env bash

## Continue to train PPO RL models using usp with sampling temperature of -1, 0.67, 0.33
## In this set of experiment, we used batchSize of 128
echo 'PPO RL training'
date +%Y,%m,%d-%H:%M:%S
hist=3
batchSz=128
for thrN in 4;
do
    for lrt in 7e-4;
    do
        for sfxTem in -1 0.67 0.33;
        do
            if [ $sfxTem -eq -1 ];
            then
                th rlMain.lua -async PPO -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize ${batchSz} -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*250000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 1500 -valSteps 2500 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_batSz${batchSz}_ppoEp2 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
                wait
            else
                th rlMain.lua -async PPO -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize ${batchSz} -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*250000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 1500 -valSteps 2500 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_batSz${batchSz}_ppoEp2 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
                wait
            fi
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
       done
    done
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S