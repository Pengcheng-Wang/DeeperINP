#!/usr/bin/env bash

## Train rl models with different # of threads
echo 'RL training test'
date +%Y,%m,%d-%H:%M:%S
hist=2
for thrN in 2 4 6;
do
    for lrt in 7e-4 2e-4 1e-3;
    do
        th rlMain.lua -async PPO -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps 1000000 -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_ppo_tune_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0 -async_valErr_coef 1 -ppo_optim_epo 1 -ciGroup2rwd 0 > /dev/null &
        th rlMain.lua -async A3C -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps 1000000 -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_a3c_tune_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0 -async_valErr_coef 1 -ppo_optim_epo 1 -ciGroup2rwd 0 > /dev/null &
        th rlMain.lua -async OneStepQ -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps 1000000 -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_dqn_tune_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0 -async_valErr_coef 1 -ppo_optim_epo 1 -ciGroup2rwd 0 > /dev/null &
        wait
    done
    echo 'done with thrN ' ${thrN}
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S
