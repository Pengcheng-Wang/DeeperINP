#!/usr/bin/env bash

## Continue to train PPO RL models using usp with sampling temperature of -1, 0.67, 0.33
echo 'DQN RL training with action temperature'
date +%Y,%m,%d-%H:%M:%S
hist=3
for actSfxTem in 0.8 0.6 0.5;
do
    for sfxTem in -1 0.5;
    do
        for seedSet in 2;
        do
            for thrN in 4;
            do
                for lrt in 7e-4;
                do
                    th rlMain.lua -async OneStepQ -network last -seed ${seedSet} -ppo_clip_thr 0.1 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*120000)) -tau 4 -memSize 18000 -epsilonSteps 90000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCI/ci_dqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet} -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem} > /dev/null &
                    wait
                    echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', RwdsfxTem: ' ${sfxTem} ', ActsfxTem: ' ${actSfxTem}
                done
            done
        done
    done
done
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S