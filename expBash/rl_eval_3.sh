#!/usr/bin/env bash

## Evaluation
### Train PPO RL models using usp with sampling temperature of 0.33 0.2 0.45. Evaluation
#echo 'DQN RL training test'
#date +%Y,%m,%d-%H:%M:%S
#hist=3
#for thrN in 4;
#do
#    for lrt in 7e-4;
#    do
#        for sfxTem in 0.67;
#        do
#            th rlMain.lua -async OneStepQ -ppo_clip_thr 0.15 -mode eval -evaTrajs 10000 -ciuTType train -network experiments/ci_dqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}/best.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*1000000)) -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_dqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_eval_bestOn1 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp -1 #${sfxTem}
#            wait
#            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
#        done
#    done
#done
#wait
#echo 'Done training'
#date +%Y,%m,%d-%H:%M:%S


## Evaluation
### Train PPO RL models using usp with sampling temperature of 0.33 0.2 0.45. Evaluation
#echo 'DQN RL training test'
#date +%Y,%m,%d-%H:%M:%S
#hist=3
#for thrN in 4;
#do
#    for lrt in 7e-4;
#    do
#        for sfxTem in 0.33;
#        do
#            th rlMain.lua -async OneStepQ -ppo_clip_thr 0.15 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/ci_dqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}/best.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*1000000)) -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_dqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_eval_bestOn1 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp -1 #${sfxTem}
#            wait
#            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
#        done
#    done
#done
#wait
#echo 'Done training'
#date +%Y,%m,%d-%H:%M:%S


## Evaluation
### Train PPO RL models using usp with sampling temperature of 0.33 0.2 0.45. Evaluation
#echo 'DQN RL training test'
#date +%Y,%m,%d-%H:%M:%S
#hist=3
#for thrN in 4;
#do
#    for lrt in 7e-4;
#    do
#        for sfxTem in 0.33;
#        do
#            th rlMain.lua -async OneStepQ -ppo_clip_thr 0.15 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/ci_dqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}/277633_0.66912.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*1000000)) -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_dqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_eval_277633_0.66912.weights.t7On1 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp -1 #${sfxTem}
#            wait
#            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
#        done
#    done
#done
#wait
#echo 'Done training'
#date +%Y,%m,%d-%H:%M:%S



# Evaluation
## Train PPO RL models using usp with sampling temperature of 0.33 0.2 0.45. Evaluation
echo 'DQN RL training test'
date +%Y,%m,%d-%H:%M:%S
hist=3
for thrN in 4;
do
    for lrt in 7e-4;
    do
        for sfxTem in 0.33;
        do
            th rlMain.lua -async OneStepQ -ppo_clip_thr 0.15 -mode eval -evaTrajs 1500 -ciuTType train -network experiments/ci_dqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}/420765_0.67766.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*1000000)) -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_dqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_eval_420765_0.67766.weights.t7_On1 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp -1 #${sfxTem}
            wait
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
        done
    done
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S