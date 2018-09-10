#!/usr/bin/env bash

## Sept 9, 2018, by pwang8
## This is the evaluation of DuelDQN RL agents in 2 fold training, full set evaluation setting. training on 1st, test on full
 ## Evaluation
 echo 'Train on 2fold1, test on full. DuelDQN RL training t1ef evaluation'
 date +%Y,%m,%d-%H:%M:%S
 hist=3
 seedSet=2
 actSfxTem=1
 ppoEpc=2
 for thrN in 4;
 do
     for lrt in 7e-4;
     do
         for sfxTem in -1;
         do
             th rlMain.lua -ac_greedy false -async OneStepQ -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/2foldCI/2fold1_ci_DuelDqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem0.8_seed${seedSet}/343175_0.84528.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ true -duel true -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 150000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 2foldCIEval_Baseline/Train2f1EvlFul/ci_DuelDqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/343175_0.84528 -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
             wait
             echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
         done
     done
 done
 wait
 echo 'Done eval'
 date +%Y,%m,%d-%H:%M:%S


 ## Evaluation
 echo 'Train on 2fold1, test on full. DuelDQN RL training t1ef evaluation'
 date +%Y,%m,%d-%H:%M:%S
 hist=3
 seedSet=2
 actSfxTem=1
 ppoEpc=2
 for thrN in 4;
 do
     for lrt in 7e-4;
     do
         for sfxTem in -1;
         do
             th rlMain.lua -ac_greedy false -async OneStepQ -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/2foldCI/2fold1_ci_DuelDqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem0.8_seed${seedSet}/746921_0.83012.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ true -duel true -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 150000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 2foldCIEval_Baseline/Train2f1EvlFul/ci_DuelDqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/746921_0.83012 -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
             wait
             echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
         done
     done
 done
 wait
 echo 'Done eval'
 date +%Y,%m,%d-%H:%M:%S


 ## Evaluation
 echo 'Train on 2fold1, test on full. DuelDQN RL training t1ef evaluation'
 date +%Y,%m,%d-%H:%M:%S
 hist=3
 seedSet=2
 actSfxTem=1
 ppoEpc=2
 for thrN in 4;
 do
     for lrt in 7e-4;
     do
         for sfxTem in -1;
         do
             th rlMain.lua -ac_greedy false -async OneStepQ -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/2foldCI/2fold1_ci_DuelDqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem0.8_seed${seedSet}/796190_0.82642.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ true -duel true -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 150000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 2foldCIEval_Baseline/Train2f1EvlFul/ci_DuelDqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/796190_0.82642 -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
             wait
             echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
         done
     done
 done
 wait
 echo 'Done eval'
 date +%Y,%m,%d-%H:%M:%S


## Let me try some Train on 2fold2, test on full. DuelDQN RL training t1ef evaluation. 9/9/2018, by pwang8
 ## Evaluation
 echo 'Train on 2fold2, test on full. DuelDQN RL training t1ef evaluation'
 date +%Y,%m,%d-%H:%M:%S
 hist=3
 seedSet=2
 actSfxTem=1
 ppoEpc=2
 for thrN in 4;
 do
     for lrt in 7e-4;
     do
         for sfxTem in -1;
         do
             th rlMain.lua -ac_greedy false -async OneStepQ -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/2foldCI/2fold2_ci_DuelDqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem0.8_seed${seedSet}/604935_0.74468.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ true -duel true -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 150000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 2foldCIEval_Baseline/Train2f2EvlFul/ci_DuelDqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/604935_0.74468 -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
             wait
             echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
         done
     done
 done
 wait
 echo 'Done eval'
 date +%Y,%m,%d-%H:%M:%S


 ## Evaluation
 echo 'Train on 2fold2, test on full. DuelDQN RL training t1ef evaluation'
 date +%Y,%m,%d-%H:%M:%S
 hist=3
 seedSet=2
 actSfxTem=1
 ppoEpc=2
 for thrN in 4;
 do
     for lrt in 7e-4;
     do
         for sfxTem in -1;
         do
             th rlMain.lua -ac_greedy false -async OneStepQ -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/2foldCI/2fold2_ci_DuelDqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem0.8_seed${seedSet}/755272_0.73077.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ true -duel true -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 150000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 2foldCIEval_Baseline/Train2f2EvlFul/ci_DuelDqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/755272_0.73077 -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
             wait
             echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
         done
     done
 done
 wait
 echo 'Done eval'
 date +%Y,%m,%d-%H:%M:%S


 ## Evaluation
 echo 'Train on 2fold2, test on full. DuelDQN RL training t1ef evaluation'
 date +%Y,%m,%d-%H:%M:%S
 hist=3
 seedSet=2
 actSfxTem=1
 ppoEpc=2
 for thrN in 4;
 do
     for lrt in 7e-4;
     do
         for sfxTem in -1;
         do
             th rlMain.lua -ac_greedy false -async OneStepQ -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/2foldCI/2fold2_ci_DuelDqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem0.8_seed${seedSet}/763376_0.72924.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ true -duel true -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*200000)) -tau 4 -memSize 18000 -epsilonSteps 150000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 2foldCIEval_Baseline/Train2f2EvlFul/ci_DuelDqn_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/763376_0.72924 -async_valErr_coef 1 -ppo_optim_epo 3 -ciGroup2rwd 0 -ciRwdStMxTemp ${sfxTem} -ciActStMxTemp ${actSfxTem}
             wait
             echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
         done
     done
 done
 wait
 echo 'Done eval'
 date +%Y,%m,%d-%H:%M:%S