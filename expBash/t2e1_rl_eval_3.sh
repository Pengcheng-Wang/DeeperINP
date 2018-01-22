#!/usr/bin/env bash

# Evaluation
echo '2fold A3C RL eval'
date +%Y,%m,%d-%H:%M:%S
hist=3
seedSet=2
for thrN in 4;
do
    for lrt in 7e-4;
    do
        for sfxTem in -1;
        do
            th rlMain.lua -ac_greedy false -async A3C -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/2foldCI/2fold1_ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_seed${seedSet}/314913_0.67965.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold2_rhnHd3RD3St2k.t7 -uspFile usp_2fold2_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCIEvalAG/Train1Evl2/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_seed${seedSet}_eval/314913_0.67965 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
            th rlMain.lua -ac_greedy true -async A3C -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/2foldCI/2fold1_ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_seed${seedSet}/314913_0.67965.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold2_rhnHd3RD3St2k.t7 -uspFile usp_2fold2_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCIEvalAG/Train1Evl2/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_seed${seedSet}_deta_eval/314913_0.67965 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
            wait
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
        done
    done
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S



# Evaluation
echo '2fold A3C RL eval'
date +%Y,%m,%d-%H:%M:%S
hist=3
seedSet=2
for thrN in 4;
do
    for lrt in 7e-4;
    do
        for sfxTem in -1;
        do
            th rlMain.lua -ac_greedy false -async A3C -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/2foldCI/2fold1_ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_seed${seedSet}/322773_0.66520.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold2_rhnHd3RD3St2k.t7 -uspFile usp_2fold2_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCIEvalAG/Train1Evl2/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_seed${seedSet}_eval/322773_0.66520 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
            th rlMain.lua -ac_greedy true -async A3C -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/2foldCI/2fold1_ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_seed${seedSet}/322773_0.66520.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold2_rhnHd3RD3St2k.t7 -uspFile usp_2fold2_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCIEvalAG/Train1Evl2/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_seed${seedSet}_deta_eval/322773_0.66520 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
            wait
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
        done
    done
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S




# Evaluation
echo '2fold A3C RL eval'
date +%Y,%m,%d-%H:%M:%S
hist=3
seedSet=2
for thrN in 4;
do
    for lrt in 7e-4;
    do
        for sfxTem in -1;
        do
            th rlMain.lua -ac_greedy false -async A3C -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/2foldCI/2fold1_ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_seed${seedSet}/337919_0.66814.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold2_rhnHd3RD3St2k.t7 -uspFile usp_2fold2_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCIEvalAG/Train1Evl2/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_seed${seedSet}_eval/337919_0.66814 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
            th rlMain.lua -ac_greedy true -async A3C -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/2foldCI/2fold1_ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_seed${seedSet}/337919_0.66814.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 200000 -valFreq 6000 -valSteps 5000 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold2_rhnHd3RD3St2k.t7 -uspFile usp_2fold2_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCIEvalAG/Train1Evl2/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_sfxTem${sfxTem}_seed${seedSet}_deta_eval/337919_0.66814 -async_valErr_coef 1 -ppo_optim_epo 2 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} > /dev/null &
            wait
            echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
        done
    done
done
wait
echo 'Done training'
date +%Y,%m,%d-%H:%M:%S