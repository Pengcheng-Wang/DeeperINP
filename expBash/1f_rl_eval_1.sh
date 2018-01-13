#!/usr/bin/env bash

# Evaluation
## Train PPO RL models using usp with sampling temperature of 0.33 0.2 0.45. Evaluation
echo '1fold PPO RL training test'
date +%Y,%m,%d-%H:%M:%S
hist=3
seedSet=2
actSfxTem=0.8
ppoEpc=2
for thrN in 4;
do
    for lrt in 7e-4;
    do
        for sfxTem in -1;
        do
            th rlMain.lua -ac_greedy false -async PPO -seed 3 -ppo_clip_thr 0.1 -mode eval -evaTrajs 1500 -ciuTType train -network experiments/1foldCI/ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_ppoEp${ppoEpc}_seed${seedSet}/157295_0.71777.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*120000)) -tau 4 -memSize 18000 -epsilonSteps 90000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCIEval/ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_ppoEp${ppoEpc}_seed${seedSet}/157295_0.71777 -async_valErr_coef 1 -ppo_optim_epo ${ppoEpc} -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
            wait
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
        done
    done
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S