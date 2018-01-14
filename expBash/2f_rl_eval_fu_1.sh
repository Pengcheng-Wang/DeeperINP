#!/usr/bin/env bash

# Evaluation
## Train PPO RL models using usp with sampling temperature of 0.33 0.2 0.45. Evaluation
echo '2fold PPO RL eval on full'
date +%Y,%m,%d-%H:%M:%S
hist=3
seedSet=2
for thrN in 4;
do
    for lrt in 7e-4;
    do
        for sfxTem in -1;
        do
            th rlMain.lua -ac_greedy false -async PPO -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 1800 -ciuTType train -network experiments/2foldCI/2fold1_ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2_seed${seedSet}/745751_0.68664.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 2foldCIEval/Train1EvlFul/ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2_seed${seedSet}_eval/745751_0.68664 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem}
            wait
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
        done
    done
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S
