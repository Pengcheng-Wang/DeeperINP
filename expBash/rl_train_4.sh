#!/usr/bin/env bash

### Train PPO RL models using Ep50 usp with sampling temperature of 0.33 0.2 0.45
### In this exp, we try to use relative policy in optim
#echo 'PPO RL (relative policy) training test'
#date +%Y,%m,%d-%H:%M:%S
#hist=3
#for thrN in 6;
#do
#    for lrt in 7e-4;
#    do
#        for sfxTem in 0.33 0.2 0.45;
#        do
#            th rlMain.lua -async PPO -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 64 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*1000000)) -tau 4 -memSize 18000 -epsilonSteps 1000000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23Ep50.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2_rlply -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ac_relative_plc true > /dev/null &
#            wait
#            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
#        done
#    done
#done
#wait
#echo 'Done training'
#date +%Y,%m,%d-%H:%M:%S


### Continue to train PPO RL models using usp with sampling temperature of -1, 0.67, 0.33 using relative policy
#echo 'PPO RL (relative policy) training'
#date +%Y,%m,%d-%H:%M:%S
#hist=3
#for thrN in 6;
#do
#    for lrt in 7e-4;
#    do
#        for sfxTem in -1 0.67 0.33;
#        do
#            if [ $sfxTem -eq -1 ];
#            then
#                th rlMain.lua -async PPO -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2_rlply -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ac_relative_plc true > /dev/null &
#                wait
#            else
#                th rlMain.lua -async PPO -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2_rlply -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ac_relative_plc true > /dev/null &
#                wait
#            fi
#            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
#       done
#    done
#done
#wait
#echo 'Done training'
#date +%Y,%m,%d-%H:%M:%S


### Continue to train PPO RL models using usp with sampling temperature of -1, 0.67, 0.33 using relative policy
#echo 'PPO RL (relative policy) training'
#date +%Y,%m,%d-%H:%M:%S
#hist=3
#for thrN in 4;
#do
#    for lrt in 7e-4;
#    do
#        for sfxTem in -1 0.67 0.33;
#        do
#            if [ $sfxTem -eq -1 ];
#            then
#                th rlMain.lua -async PPO -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*250000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 1500 -valSteps 2500 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2_rlply -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ac_relative_plc true -network experiments/Jan11_2018_Thr6/ci_ppo_lrt_${lrt}_thr_6_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2_rlply/best.weights.t7 > /dev/null &
#                wait
#            else
#                th rlMain.lua -async PPO -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*250000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 1500 -valSteps 2500 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2_rlply -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ac_relative_plc true -network experiments/Jan11_2018_Thr6/ci_ppo_lrt_${lrt}_thr_6_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2_rlply/best.weights.t7 > /dev/null &
#                wait
#            fi
#            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
#       done
#    done
#done
#wait
#echo 'Done training'
#date +%Y,%m,%d-%H:%M:%S


## Jan 12, 2018.
## Continue the unfinished training from Jan 11, where we stopped them. All seed setting in Setup is 1

## Continue to train PPO RL models using usp with sampling temperature of 0.67, 0.33
echo 'Contiune PPO RL training'
date +%Y,%m,%d-%H:%M:%S
hist=3
for thrN in 4;
do
    for lrt in 7e-4;
    do
        for sfxTem in 0.33;
        do
            if [ $sfxTem -eq -1 ];
            then
                th rlMain.lua -async PPO -network last -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*250000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 1500 -valSteps 2500 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
                wait
            else
                th rlMain.lua -async PPO -network last -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*250000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 1500 -valSteps 2500 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_ppo_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_ppoEp2 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
                wait
            fi
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
       done
    done
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S

## Continue to train PPO RL models using usp with sampling temperature of -1, 0.67, 0.33
echo 'Continue DQN RL training'
date +%Y,%m,%d-%H:%M:%S
hist=3
for thrN in 4;
do
    for lrt in 7e-4;
    do
        for sfxTem in 0.33;
        do
            if [ $sfxTem -eq -1 ];
            then
                th rlMain.lua -async OneStepQ -network last -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*250000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 1500 -valSteps 2500 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_dqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem} -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
                wait
            else
                th rlMain.lua -async OneStepQ -network last -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*250000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 1500 -valSteps 2500 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_dqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem} -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
                wait
            fi
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
       done
    done
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S


## Continue to train A3C RL models using usp with sampling temperature of -1, 0.67, 0.33
echo 'Continue A3C RL training'
date +%Y,%m,%d-%H:%M:%S
hist=3
for thrN in 4;
do
    for lrt in 7e-4;
    do
        for sfxTem in 0.33;
        do
            if [ $sfxTem -eq -1 ];
            then
                th rlMain.lua -async A3C -network last -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*250000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 1500 -valSteps 2500 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem} -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
                wait
            else
                th rlMain.lua -async A3C -network last -ppo_clip_thr 0.15 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*250000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 1500 -valSteps 2500 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem} -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
                wait
            fi
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
       done
    done
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S