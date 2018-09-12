#!/usr/bin/env bash

## Sept 11, 2018, by pwang8
## This is the evaluation of the DQN with experience replay model. Training on 2fold half courpus, and evaluation
## are on the other half corpus
echo 'DQN_EPR RL with Experience Replay training with action temperature, 2fold1, evaluated on 2nd half corpus'
date +%Y,%m,%d-%H:%M:%S
hist=3
for actSfxTem in 1;
do
    for sfxTem in -1;
    do
        for seedSet in 2;
        do
            for thrN in 4;
            do
                for lrt in 7e-4;
                do
                    th rlMain.lua -ciTemCnn 2 -mode eval -evaTrajs 5000 -network experiments/2foldCI/2fold1_ci_DQN_ExpRpl_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem0.8_seed${seedSet}/Epoch_2100.80.weights.t7 -seed ${seedSet} -ppo_clip_thr 0.1 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*300000)) -tau 4 -memSize 18000 -epsilonSteps 225000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold2_rhnHd3RD3St2k.t7 -uspFile usp_2fold2_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCIEval_Baseline/Train1Evl2/ci_DQN_ExpRpl_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/Epoch_2100.80 -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
                    wait
                    echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', RwdsfxTem: ' ${sfxTem} ', ActsfxTem: ' ${actSfxTem}
                done
            done
        done
    done
done
echo 'Done DQN_EPR RL with Experience Replay trained on 2fold1, evaluated on 2nd half corpus'
date +%Y,%m,%d-%H:%M:%S
wait


## Sept 11, 2018, by pwang8
## This is the evaluation of the DQN with experience replay model. Training on 2fold half courpus, and evaluation
## are on the other half corpus
echo 'DQN_EPR RL with Experience Replay training with action temperature, 2fold1, evaluated on 2nd half corpus'
date +%Y,%m,%d-%H:%M:%S
hist=3
for actSfxTem in 1;
do
    for sfxTem in -1;
    do
        for seedSet in 2;
        do
            for thrN in 4;
            do
                for lrt in 7e-4;
                do
                    th rlMain.lua -ciTemCnn 2 -mode eval -evaTrajs 5000 -network experiments/2foldCI/2fold1_ci_DQN_ExpRpl_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem0.8_seed${seedSet}/Epoch_3900.79.weights.t7 -seed ${seedSet} -ppo_clip_thr 0.1 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*300000)) -tau 4 -memSize 18000 -epsilonSteps 225000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold2_rhnHd3RD3St2k.t7 -uspFile usp_2fold2_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCIEval_Baseline/Train1Evl2/ci_DQN_ExpRpl_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/Epoch_3900.79 -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
                    wait
                    echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', RwdsfxTem: ' ${sfxTem} ', ActsfxTem: ' ${actSfxTem}
                done
            done
        done
    done
done
echo 'Done DQN_EPR RL with Experience Replay trained on 2fold1, evaluated on 2nd half corpus'
date +%Y,%m,%d-%H:%M:%S
wait


## Sept 11, 2018, by pwang8
## This is the evaluation of the DQN with experience replay model. Training on 2fold half courpus, and evaluation
## are on the other half corpus
echo 'DQN_EPR RL with Experience Replay training with action temperature, 2fold1, evaluated on 2nd half corpus'
date +%Y,%m,%d-%H:%M:%S
hist=3
for actSfxTem in 1;
do
    for sfxTem in -1;
    do
        for seedSet in 2;
        do
            for thrN in 4;
            do
                for lrt in 7e-4;
                do
                    th rlMain.lua -ciTemCnn 2 -mode eval -evaTrajs 5000 -network experiments/2foldCI/2fold1_ci_DQN_ExpRpl_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem0.8_seed${seedSet}/Epoch_4800.78.weights.t7 -seed ${seedSet} -ppo_clip_thr 0.1 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*300000)) -tau 4 -memSize 18000 -epsilonSteps 225000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold2_rhnHd3RD3St2k.t7 -uspFile usp_2fold2_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCIEval_Baseline/Train1Evl2/ci_DQN_ExpRpl_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/Epoch_4800.78 -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
                    wait
                    echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', RwdsfxTem: ' ${sfxTem} ', ActsfxTem: ' ${actSfxTem}
                done
            done
        done
    done
done
echo 'Done DQN_EPR RL with Experience Replay trained on 2fold1, evaluated on 2nd half corpus'
date +%Y,%m,%d-%H:%M:%S
wait


## Sept 11, 2018, by pwang8
## This is the evaluation of the DQN with experience replay model. Training on 2fold2 half courpus, and evaluation
## are on the other half corpus
echo 'DQN_EPR RL with Experience Replay training with action temperature, 2fold2, evaluated on 1st half corpus'
date +%Y,%m,%d-%H:%M:%S
hist=3
for actSfxTem in 1;
do
    for sfxTem in -1;
    do
        for seedSet in 2;
        do
            for thrN in 4;
            do
                for lrt in 7e-4;
                do
                    th rlMain.lua -ciTemCnn 2 -mode eval -evaTrajs 5000 -network experiments/2foldCI/2fold2_ci_DQN_ExpRpl_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem0.8_seed${seedSet}/Epoch_2100.68.weights.t7 -seed ${seedSet} -ppo_clip_thr 0.1 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*300000)) -tau 4 -memSize 18000 -epsilonSteps 225000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold1_rhnHd3RD3St2k.t7 -uspFile usp_2fold1_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCIEval_Baseline/Train2Evl1/ci_DQN_ExpRpl_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/Epoch_2100.68 -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
                    wait
                    echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', RwdsfxTem: ' ${sfxTem} ', ActsfxTem: ' ${actSfxTem}
                done
            done
        done
    done
done
echo 'Done DQN_EPR RL with Experience Replay trained on 2fold2, evaluated on 1st half corpus'
date +%Y,%m,%d-%H:%M:%S
wait


## Sept 11, 2018, by pwang8
## This is the evaluation of the DQN with experience replay model. Training on 2fold2 half courpus, and evaluation
## are on the other half corpus
echo 'DQN_EPR RL with Experience Replay training with action temperature, 2fold2, evaluated on 1st half corpus'
date +%Y,%m,%d-%H:%M:%S
hist=3
for actSfxTem in 1;
do
    for sfxTem in -1;
    do
        for seedSet in 2;
        do
            for thrN in 4;
            do
                for lrt in 7e-4;
                do
                    th rlMain.lua -ciTemCnn 2 -mode eval -evaTrajs 5000 -network experiments/2foldCI/2fold2_ci_DQN_ExpRpl_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem0.8_seed${seedSet}/Epoch_2700.66.weights.t7 -seed ${seedSet} -ppo_clip_thr 0.1 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*300000)) -tau 4 -memSize 18000 -epsilonSteps 225000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold1_rhnHd3RD3St2k.t7 -uspFile usp_2fold1_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCIEval_Baseline/Train2Evl1/ci_DQN_ExpRpl_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/Epoch_2700.66 -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
                    wait
                    echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', RwdsfxTem: ' ${sfxTem} ', ActsfxTem: ' ${actSfxTem}
                done
            done
        done
    done
done
echo 'Done DQN_EPR RL with Experience Replay trained on 2fold2, evaluated on 1st half corpus'
date +%Y,%m,%d-%H:%M:%S
wait


## Sept 11, 2018, by pwang8
## This is the evaluation of the DQN with experience replay model. Training on 2fold2 half courpus, and evaluation
## are on the other half corpus
echo 'DQN_EPR RL with Experience Replay training with action temperature, 2fold2, evaluated on 1st half corpus'
date +%Y,%m,%d-%H:%M:%S
hist=3
for actSfxTem in 1;
do
    for sfxTem in -1;
    do
        for seedSet in 2;
        do
            for thrN in 4;
            do
                for lrt in 7e-4;
                do
                    th rlMain.lua -ciTemCnn 2 -mode eval -evaTrajs 5000 -network experiments/2foldCI/2fold2_ci_DQN_ExpRpl_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem0.8_seed${seedSet}/Epoch_4800.65.weights.t7 -seed ${seedSet} -ppo_clip_thr 0.1 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*300000)) -tau 4 -memSize 18000 -epsilonSteps 225000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold1_rhnHd3RD3St2k.t7 -uspFile usp_2fold1_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCIEval_Baseline/Train2Evl1/ci_DQN_ExpRpl_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/Epoch_4800.65 -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
                    wait
                    echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', RwdsfxTem: ' ${sfxTem} ', ActsfxTem: ' ${actSfxTem}
                done
            done
        done
    done
done
echo 'Done DQN_EPR RL with Experience Replay trained on 2fold2, evaluated on 1st half corpus'
date +%Y,%m,%d-%H:%M:%S
wait