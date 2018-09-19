#!/usr/bin/env bash

## Sept 19, 2018, by pwang8
## This is the experiment of linear RL agent training without extra help of the stabilization techniques derived and utilized in deep RL methods
echo 'Linear RL (no stabilization) training with action temperature, 1fold'
date +%Y,%m,%d-%H:%M:%S
hist=3
for actSfxTem in 0.8;
do
    for sfxTem in -1;
    do
        for seedSet in 2;
        do
            for thrN in 1;
            do
                for lrt in 7e-4;
                do
                    th rlMain.lua -async OneStepQ -ciTemCnn 0 -rlnnLinear true -network last -seed ${seedSet} -ppo_clip_thr 0.1 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 1 -memSize 100 -epsilonSteps 150000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCI/ci_linearRL_noStb_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet} -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem} > /dev/null &
                    wait
                    echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', RwdsfxTem: ' ${sfxTem} ', ActsfxTem: ' ${actSfxTem}
                done
            done
        done
    done
done
echo 'Done Linear (no stabilization) 1fold'
date +%Y,%m,%d-%H:%M:%S
wait


echo 'Linear RL (no stabilization) training with action temperature, 2fold'
date +%Y,%m,%d-%H:%M:%S
hist=3
for actSfxTem in 0.8;
do
    for sfxTem in -1;
    do
        for seedSet in 2;
        do
            for thrN in 1;
            do
                for lrt in 7e-4;
                do
                    th rlMain.lua -async OneStepQ -ciTemCnn 0 -rlnnLinear true -network last -seed ${seedSet} -ppo_clip_thr 0.1 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 1 -memSize 100 -epsilonSteps 150000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold1_rhnHd3RD3St2k.t7 -uspFile usp_2fold1_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCI/2fold1_ci_linearRL_noStb_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet} -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem} > /dev/null &
                    wait
                    echo 'done with fold 1, thrN: ' ${thrN} ', lrt: ' ${lrt} ', RwdsfxTem: ' ${sfxTem} ', ActsfxTem: ' ${actSfxTem}
                    th rlMain.lua -async OneStepQ -ciTemCnn 0 -rlnnLinear true -network last -seed ${seedSet} -ppo_clip_thr 0.1 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 1 -memSize 100 -epsilonSteps 150000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_2fold2_rhnHd3RD3St2k.t7 -uspFile usp_2fold2_cnnmoe_L2H2K1E30e23_Ep1k5.t7 -_id 2foldCI/2fold2_ci_linearRL_noStb_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet} -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem} > /dev/null &
                    wait
                    echo 'done with fold 2, thrN: ' ${thrN} ', lrt: ' ${lrt} ', RwdsfxTem: ' ${sfxTem} ', ActsfxTem: ' ${actSfxTem}
                done
            done
        done
    done
done
echo 'Done Linear (no stabilization) 2fold'
date +%Y,%m,%d-%H:%M:%S

