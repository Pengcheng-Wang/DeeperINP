#!/usr/bin/env bash

##################################################################
###### The following exp add random act and score
##################################################################

# Evaluation
## Train PPO RL models using usp with sampling temperature of 0.33 0.2 0.45. Evaluation
echo '1fold PPO RL training test Oh'
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
            th rlMain.lua -ac_greedy true -async PPO -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 3000 -ciuTType train -network experiments/1foldCI/ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_ppoEp${ppoEpc}_seed${seedSet}/474724_0.64236.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*120000)) -tau 4 -memSize 18000 -epsilonSteps 90000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCIEval/ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_ppoEp${ppoEpc}_seed${seedSet}_deta_Rnd33/474724_0.64236 -async_valErr_coef 1 -ppo_optim_epo ${ppoEpc} -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem} -ciActRndSmp 0.33 -ciRwdRndSmp 0.33
            wait
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
        done
    done
done
wait
echo 'Done training 1'
date +%Y,%m,%d-%H:%M:%S

echo '1fold PPO RL training test Cool'
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
            th rlMain.lua -ac_greedy true -async PPO -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 3000 -ciuTType train -network experiments/1foldCI/ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_ppoEp${ppoEpc}_seed${seedSet}/474724_0.64236.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*120000)) -tau 4 -memSize 18000 -epsilonSteps 90000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCIEval/ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_ppoEp${ppoEpc}_seed${seedSet}_deta_Rnd20/474724_0.64236 -async_valErr_coef 1 -ppo_optim_epo ${ppoEpc} -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem} -ciActRndSmp 0.2 -ciRwdRndSmp 0.2
            wait
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
        done
    done
done
wait
echo 'Done training 1'
date +%Y,%m,%d-%H:%M:%S


echo '1fold PPO RL training test 175'
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
            th rlMain.lua -ac_greedy true -async PPO -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 3000 -ciuTType train -network experiments/1foldCI/ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_ppoEp${ppoEpc}_seed${seedSet}/478552_0.67347.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*120000)) -tau 4 -memSize 18000 -epsilonSteps 90000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCIEval/ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_ppoEp${ppoEpc}_seed${seedSet}_deta_Rnd20/478552_0.67347 -async_valErr_coef 1 -ppo_optim_epo ${ppoEpc} -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem} -ciActRndSmp 0.2 -ciRwdRndSmp 0.2
            wait
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
        done
    done
done
wait
echo 'Done training 2'
date +%Y,%m,%d-%H:%M:%S


echo '1fold PPO RL training test 232'
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
            th rlMain.lua -ac_greedy true -async PPO -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 3000 -ciuTType train -network experiments/1foldCI/ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_ppoEp${ppoEpc}_seed${seedSet}/478552_0.67347.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*120000)) -tau 4 -memSize 18000 -epsilonSteps 90000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCIEval/ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_ppoEp${ppoEpc}_seed${seedSet}_deta_Rnd33/478552_0.67347 -async_valErr_coef 1 -ppo_optim_epo ${ppoEpc} -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem} -ciActRndSmp 0.2 -ciRwdRndSmp 0.33
            wait
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
        done
    done
done
wait
echo 'Done training 2'
date +%Y,%m,%d-%H:%M:%S