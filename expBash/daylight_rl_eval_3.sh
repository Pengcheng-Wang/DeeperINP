#!/usr/bin/env bash

######################################################################
###### This set of experiments evaluate A3C model performance trained
###### on more deterministic simulator and interact with more stochastic
###### simulator
######################################################################

## Stochastic policy for A3C
echo '1fold A3C RL Eval of deterministic model interacting with stochastic sims, 1st'
date +%Y,%m,%d-%H:%M:%S
hist=3
seedSet=2
actSfxTem=0.8
ppoEpc=2
for thrN in 4;
do
 for lrt in 7e-4;
 do
     for sfxTem in 0.5;
     do
         th rlMain.lua -ac_greedy false -async A3C -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/1foldCI/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/175354_0.73448.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*120000)) -tau 4 -memSize 18000 -epsilonSteps 90000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCIEvalOnStc1/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}_onAct1Rwd1/175354_0.73448 -async_valErr_coef 1 -ppo_optim_epo ${ppoEpc} -ciGroup2rwd 0 -ciRwdStMxTemp 1 -ciActStMxTemp 1
#         wait
         echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
     done
 done
done
#wait
echo 'Ending A3C RL eval, 28210'
date +%Y,%m,%d-%H:%M:%S


echo '1fold A3C RL Eval of deterministic model interacting with stochastic sims, 2nd'
date +%Y,%m,%d-%H:%M:%S
hist=3
seedSet=2
actSfxTem=0.8
ppoEpc=2
for thrN in 4;
do
 for lrt in 7e-4;
 do
     for sfxTem in 0.5;
     do
         th rlMain.lua -ac_greedy false -async A3C -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/1foldCI/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/293616_0.73776.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*120000)) -tau 4 -memSize 18000 -epsilonSteps 90000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCIEvalOnStc1/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}_onAct1Rwd1/293616_0.73776 -async_valErr_coef 1 -ppo_optim_epo ${ppoEpc} -ciGroup2rwd 0 -ciRwdStMxTemp 1 -ciActStMxTemp 1
#         wait
         echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
     done
 done
done
#wait
echo 'Ending A3C RL eval, 28210'
date +%Y,%m,%d-%H:%M:%S


echo '1fold A3C RL Eval of deterministic model interacting with stochastic sims, 3rd'
date +%Y,%m,%d-%H:%M:%S
hist=3
seedSet=2
actSfxTem=0.8
ppoEpc=2
for thrN in 4;
do
 for lrt in 7e-4;
 do
     for sfxTem in 0.5;
     do
         th rlMain.lua -ac_greedy false -async A3C -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/1foldCI/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/235422_0.71731.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*120000)) -tau 4 -memSize 18000 -epsilonSteps 90000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCIEvalOnStc1/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}_onAct1Rwd1/235422_0.71731 -async_valErr_coef 1 -ppo_optim_epo ${ppoEpc} -ciGroup2rwd 0 -ciRwdStMxTemp 1 -ciActStMxTemp 1
#         wait
         echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
     done
 done
done
wait
echo 'Ending A3C RL eval, 28210'
date +%Y,%m,%d-%H:%M:%S


## Deterministic policy for A3C
echo '1fold A3C RL Eval of deterministic model interacting with deterministic sims, 1st'
date +%Y,%m,%d-%H:%M:%S
hist=3
seedSet=2
actSfxTem=0.8
ppoEpc=2
for thrN in 4;
do
 for lrt in 7e-4;
 do
     for sfxTem in 0.5;
     do
         th rlMain.lua -ac_greedy true -async A3C -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/1foldCI/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/175354_0.73448.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*120000)) -tau 4 -memSize 18000 -epsilonSteps 90000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCIEvalOnStc1/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}_deta_onAct1Rwd1/175354_0.73448 -async_valErr_coef 1 -ppo_optim_epo ${ppoEpc} -ciGroup2rwd 0 -ciRwdStMxTemp 1 -ciActStMxTemp 1
#         wait
         echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
     done
 done
done
#wait
echo 'Ending A3C RL eval, 28210'
date +%Y,%m,%d-%H:%M:%S


echo '1fold A3C RL Eval of deterministic model interacting with deterministic sims, 2nd'
date +%Y,%m,%d-%H:%M:%S
hist=3
seedSet=2
actSfxTem=0.8
ppoEpc=2
for thrN in 4;
do
 for lrt in 7e-4;
 do
     for sfxTem in 0.5;
     do
         th rlMain.lua -ac_greedy true -async A3C -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/1foldCI/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/293616_0.73776.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*120000)) -tau 4 -memSize 18000 -epsilonSteps 90000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCIEvalOnStc1/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}_deta_onAct1Rwd1/293616_0.73776 -async_valErr_coef 1 -ppo_optim_epo ${ppoEpc} -ciGroup2rwd 0 -ciRwdStMxTemp 1 -ciActStMxTemp 1
#         wait
         echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
     done
 done
done
#wait
echo 'Ending A3C RL eval, 28210'
date +%Y,%m,%d-%H:%M:%S


echo '1fold A3C RL Eval of deterministic model interacting with deterministic sims, 3rd'
date +%Y,%m,%d-%H:%M:%S
hist=3
seedSet=2
actSfxTem=0.8
ppoEpc=2
for thrN in 4;
do
 for lrt in 7e-4;
 do
     for sfxTem in 0.5;
     do
         th rlMain.lua -ac_greedy true -async A3C -seed ${seedSet} -ppo_clip_thr 0.1 -mode eval -evaTrajs 5000 -ciuTType train -network experiments/1foldCI/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}/235422_0.71731.weights.t7 -eta ${lrt} -momentum 0.99 -bootstraps 0 -batchSize 32 -hiddenSize 32 -doubleQ false -duel false -recurrent false -optimiser adam -threads ${thrN} -steps $((thrN*120000)) -tau 4 -memSize 18000 -epsilonSteps 90000 -valFreq 1500 -valSteps 2100 -bootstraps 0 -PALpha 0 -entropyBeta 0.001 -lstmHist 10 -rnnHdSizeL1 21 -rnnHdLyCnt 3 -lstmHistUsp 2 -rnnHdSizeL1Usp 21 -rnnHdLyCntUsp 2 -learnStart 2000 -progFreq 1000 -histLen ${hist} -asyncOptimFreq 1 -ciunet rlLoad -uSimScSoft 1 -uppModel rnn_rhn -uppModelUsp cnn_moe -uapFile uap_rhnL3RD3.t7 -uspFile usp_cnnmoe_L2H2K1E30e23.t7 -_id 1foldCIEvalOnStc1/ci_a3c_lrt_${lrt}_thr_${thrN}_hist${hist}_rwd0_RwdTem${sfxTem}_ActTem${actSfxTem}_seed${seedSet}_deta_onAct1Rwd1/235422_0.71731 -async_valErr_coef 1 -ppo_optim_epo ${ppoEpc} -ciGroup2rwd 0 -ciRwdStMxTemp 1 -ciActStMxTemp 1
#         wait
         echo 'done with thrN: ' ${thrN} ', lrt: ' ${lrt} ', sfxTem: ' ${sfxTem}
     done
 done
done
wait
echo 'Ending A3C RL eval, 28210, All.'
date +%Y,%m,%d-%H:%M:%S