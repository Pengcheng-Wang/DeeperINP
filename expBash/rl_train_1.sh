#!/usr/bin/env bash

### Train rl models with different # of threads
#echo 'RL training test'
#date +%Y,%m,%d-%H:%M:%S
#hist=2
#for thrN in 2 4 6;
#do
#    for lrt in 7e-4 2e-4 1e-3;
#    do
#        th rlMain.lua -async PPO -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps 1000000 -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_ppo_tune_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0 -async_valErr_coef 1 -ppo_optim_epo 1 -ciGroup2rwd 0 > /dev/null &
#        th rlMain.lua -async A3C -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps 1000000 -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_a3c_tune_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0 -async_valErr_coef 1 -ppo_optim_epo 1 -ciGroup2rwd 0 > /dev/null &
#        th rlMain.lua -async OneStepQ -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps 1000000 -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_dqn_tune_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0 -async_valErr_coef 1 -ppo_optim_epo 1 -ciGroup2rwd 0 > /dev/null &
#        wait
#    done
#    echo 'done with thrN ' ${thrN}
#done
#wait
#echo 'Done training'
#date +%Y,%m,%d-%H:%M:%S


### Train PPO RL models using Ep50 usp with sampling temperature of 0.33 0.2 0.45
#echo 'PPO RL training test'
#date +%Y,%m,%d-%H:%M:%S
#hist=3
#for thrN in 6;
#do
#    for lrt in 7e-4;
#    do
#        for sfxTem in 0.33 0.2 0.45;
#        do
#            th rlMain.lua -async PPO -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 64 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*1000000)) -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23Ep50.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
#            wait
#            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
#        done
#    done
#done
#wait
#echo 'Done training'
#date +%Y,%m,%d-%H:%M:%S


## Evaluation
# ## Train PPO RL models using Ep50 usp with sampling temperature of 0.33 0.2 0.45. Evaluation
# echo 'PPO RL training test'
# date +%Y,%m,%d-%H:%M:%S
# hist=3
# for thrN in 6;
# do
#     for lrt in 7e-4;
#     do
#         for sfxTem in 0.33;
#         do
#             th rlMain.lua -async PPO -ppo_clip_thr 0.15 -mode eval -evaTrajs 10000 -ciuTType train -network experiments/ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2/last.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 64 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*1000000)) -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23Ep50.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem}
#             wait
#             echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
#         done
#     done
# done
# wait
# echo 'Done training'
# date +%Y,%m,%d-%H:%M:%S


## Continue to train PPO RL models using usp with sampling temperature of -1, 0.67, 0.33
echo 'PPO RL training'
date +%Y,%m,%d-%H:%M:%S
hist=3
for thrN in 6;
do
    for lrt in 7e-4;
    do
        for sfxTem in -1 0.67 0.33;
        do
            if [ $sfxTem -eq -1 ];
            then
                th rlMain.lua -async PPO -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 500000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -network experiments/Jan9_2018_citestWith1500Usp/ci_ppo_tune_lrt_7e-4_thr_4_hist3_rwd0/last996038.weights.t7 > /dev/null &
                wait
            else
                th rlMain.lua -async PPO -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 500000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
                wait
            fi
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
       done
    done
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S
