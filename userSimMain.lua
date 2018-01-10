local CIFileReader = require 'file_reader'
local CIUserSimulator = require 'UserSimulator'
local CIUserActsPredictor = require 'UserSimLearner/UserActsPredictor'
local CIUserScorePredictor = require 'UserSimLearner/UserScorePredictor'
local CIUserActScorePredictor = require 'UserSimLearner/UserActScorePredictor'
local CIUserBehaviorGenerator = require 'UserSimLearner/UserBehaviorGenerator'
local CIUserBehaviorGenEvaluator = require 'UserSimLearner/UserBehaviorGenEvaluator'
local CIUserSimEnv = require 'UserSimLearner/CIUserSimEnv'

opt = lapp[[
       --trType           (default "rl")        training type : sc (score) | ac (action) | bg (behavior generation) | rl (implement rlenvs API) | ev (evaluation of act/score prediction)
       -s,--save          (default "upplogs")   subdirectory to save logs
       -n,--ciunet        (default "")          reload pretrained CI user simulation network
       -m,--uppModel      (default "rnn_rhn")   type of model to train: moe | mlp | linear | rnn_lstm | rnn_rhn | rnn_blstm | rnn_bGridlstm | cnn_uSimTempCnn | cnn_uSimCnn_moe
       -m,--uppModelUsp   (default "cnn_moe")   type of model to train: moe | mlp | linear | rnn_lstm | rnn_rhn | rnn_blstm | rnn_bGridlstm | cnn_uSimTempCnn | cnn_uSimCnn_moe. This is used only in rl mode for evaluating player simulaiton model
       --uppModelRNNDom   (default 0)           indicator of whether the model is an RNN model and uses dropout masks from outside of the model. 0 for not using outside mask. Otherwise, this number represents the number of gates used in RNN model
       --gridLstmTieWhts  (default 1)           indicator of whether the GridLSTM will have shared, tied weights along depth dimension. 1 means with shared weights, 0 means non-shared weights
       -f,--full                                use the full dataset
       -p,--plot                                plot while training
       -o,--optimization  (default "adam")       optimization: SGD | LBFGS | adam | rmsprop
       -r,--learningRate  (default 2e-4)        learning rate, for SGD only
       -b,--batchSize     (default 30)          batch size
       -m,--momentum      (default 0.9)         momentum, for SGD only
       -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
       --coefL1           (default 0)           L1 penalty on the weights
       --coefL2           (default 0)           L2 penalty on the weights
       -t,--threads       (default 4)           number of threads
       -g,--gpu           (default 0)           gpu device id, 0 for using cpu
       --seed             (default 1)           Random seed
       --prepro           (default "std")       input state feature preprocessing: rsc | std
       --rnnHdSizeL1      (default 21)          rnn hidden layer size
       --rnnHdSizeL2      (default 0)           rnn hidden layer size in 2nd lstm layer
       --rnnHdLyCnt       (default 2)           number of rnn/cnn hidden layer. Default is 2 bcz only when rnnHdSizeL2 is not 0 this opt will be examined. The RHN and Bayesian LSTM rnn number also uses this opt param. I'm also trying to use it for CNN hidden layer counting.
       --rnnHdSizeL1Usp   (default 21)          rnn hidden layer size for usp model, only used when mode is rl, for purpose of player simulator evaluation
       --rnnHdSizeL2Usp   (default 0)           rnn hidden layer size in 2nd lstm layer for usp model, only used when mode is rl, for purpose of player simulator evaluation
       --rnnHdLyCntUsp    (default 2)           number of rnn/cnn hidden layer. for usp model, only used when mode is rl, for purpose of player simulator evaluation
       --rhnReccDept      (default 5)           The recurrent depth of RHN model in one layer
       --rnnResidual      (default 0)           Whether apply residual connection in RNN player simulation models. 0 to turn if off, 1 to turn it on.
       --lstmHist         (default 10)          lstm hist length. This influence the rnn tensor table construction in data preparation. Attention: we also use it as history length indicator (input frame number) in CNN models
       --lstmHistUsp      (default 2)           lstm hist length for User score predictor. This is only used when mode is rl, for the purpose of evaluating player simulater
       --cnnKernelWidth   (default 3)           Temporal Convolution kernel width
       --cnnConnType      (default "v4")        Residual connection type in player simulation CNN model
       --moeExpCnt        (default 32)          Number of expert modules used in moe model
       --uSimGru          (default 0)           whether to substitue lstm with gru (0 for using lstm, 1 for GRU)
       --uSimLstmBackLen  (default 3)           The maximum step applied in bptt in lstm
       --ubgDir           (default "ubgModel")  directory storing uap and usp models
       --uapFile          (default "uap.t7")    file storing userActsPredictor model
       --uspFile          (default "usp.t7")    file storing userScorePredictor model
       --actSmpLen        (default 8)           The sampling candidate list length for user action generation
       --ciuTType         (default "train")     Training or testing or validation for use sim model train | test | train_tr
       --actEvaScp        (default 1)           The action selection range in prediction evaluation calculation, corresponds to the top-i prediction accuracy
       --actSmpEps        (default 0)           User action sampling threshold. If rand se than this value, reture 1st pred. Otherwise, sample sim user's next action according to the predicted distribution
       --rwdSmpEps        (default 0)           User reward sampling threshold. If rand se than this value, reture 1st pred. Otherwise, sample sim user's predicted outcome according to the predicted distribution
       --uSimShLayer      (default 0)           Whether the lower layers in Action and Score prediction NNs are shared. If this value is 1, use shared layers
       --uSimScSoft       (default 0)           The criterion weight of the score regression module in UserScoreSoftPrediction model. The value of this param should be in [0,1]. When it is 0, Soft prediction is off, and UserScorePrediction script is utilized
       --uSimBayesEvl     (default 0)           The Bayesian sampling method to do user simulation model evaluation. When the value is 0, no Bayesian sampling is conducted. Otherwise, it will be conducted to that number of times
       --rlEvnIte         (default 10000)       No of iterations in rl type of evaluation
       --ciGroup2rwd      (default -1)          Reward signal design at terminal for 2nd group (below nlg median). It can be either 0 or -1
       --ciRwdStMxTemp    (default -1)          The temperature hyper-param used in Softmax distribution re-approximation. This is used in Reward sampling. If this re-approximation is not used, and random sampling is used, this param should be set to -1
       --usimTrIte        (default 400)         No of iterations used in user simulation model training. Recom for act training is 300, score training is 3000
       --termActSmgLen    (default 50)          The length above which user termination action would be highly probably sampled. The observed avg length is about 40
       --termActSmgEps    (default 0.9)         The probability which user termination action would be sampled after certain length
       --testSetDivSeed   (default 2)           The default seed value when separating a test set from the dataset
       --validSetDivSeed  (default 3)           The default seed value when separating a validation set out from the training set
       --trainTwoFoldSim  (default 0)           If this item is 1, we train player simulation model using 50% of data, meaning constructing player sim model for 2-fold cross validation in DRL evaluation
       --dropoutUSim      (default 0.1)         The dropout rate used in user simulation model building. Set 0 to turn off droput
       --testOnTestFreq   (default 1)           The frequency of testing user simulation model's performance on test/train_valid set
       --testOnTestSoftScoreFreq   (default 30) The frequency of testing soft user outcome model's performance on test/train_valid set
       --scorePredStateScope     (default 60)   The range of distance of a player state to the end of the player interaction trajectory (ending state) that makes the state a valid training data poing for score prediction. The idea is to throw out early interaction state for user outcome predictor training
       --actPredDataAug   (default 1)           Whether to use data augmentation in action prediction model training. 0 for not using, 1 for using.
    ]]

-- threads and default tensor type
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
-- Set manual seed
torch.manualSeed(opt.seed)

-- set the uppModelRNNDom indicator in opt, which indicates whether the model is an RNN model, and uses dropout mask from outside the model construction
-- right now, the rhn model, and Bayesian lstm model (following Gal's implementation), and GridLSTM model use outside dropout mask
if string.sub(opt.uppModel, 1, 7) == 'rnn_rhn' then
    -- rnn_rhn uses double-sized dropout mask to drop out inputs of calculation of t-gate and transformed inner cell state
    opt.uppModelRNNDom = 2
elseif string.sub(opt.uppModel, 1, 9) == 'rnn_blstm' or string.sub(opt.uppModel, 1, 13) == 'rnn_bGridlstm' then
    -- lstm used quad-sized dropout mask to drop out inputs of calculation of the 3 gates and transformed inner cell state
    opt.uppModelRNNDom = 4
else
    opt.uppModelRNNDom = 0
end

-- Read CI trace and survey data files, and do validation
local fr = CIFileReader()
fr:evaluateTraceFile()
fr:evaluateSurveyData()

-- Construct CI user simulator model using real user data
local CIUserModel = CIUserSimulator(fr, opt)

assert(opt.uSimScSoft >=0 and opt.uSimScSoft <= 1, 'opt.uSimScSoft should range in [0,1]')
if opt.trType == 'sc' and opt.uSimShLayer < 0.5 and opt.uSimScSoft == 0 then
    local CIUserScorePred = CIUserScorePredictor(CIUserModel, opt)
    for i=1, opt.usimTrIte do
        CIUserScorePred:trainOneEpoch()
    end
elseif opt.trType == 'sc' and opt.uSimShLayer < 0.5 and opt.uSimScSoft > 0 then
    local CIUserScoreSoftPredictor = require 'UserSimLearner/UserScoreSoftPredictor'
    local CIUserScoreSoftPred = CIUserScoreSoftPredictor(CIUserModel, opt)
    for i=1, opt.usimTrIte do
        CIUserScoreSoftPred:trainOneEpoch()
    end
elseif opt.trType == 'ac' and opt.uSimShLayer < 0.5 then
    local CIUserActsPred = CIUserActsPredictor(CIUserModel, opt)
    for i=1, opt.usimTrIte do
        CIUserActsPred:trainOneEpoch()
    end
elseif (opt.trType == 'ac' or opt.trType == 'sc') and opt.uSimShLayer > 0.5 then
    local CIUserActScorePred = CIUserActScorePredictor(CIUserModel, opt)
    for i=1, opt.usimTrIte do
        CIUserActScorePred:trainOneEpoch()
    end
elseif opt.trType == 'rl' then

    local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available. Needed in CIUserSimEnv
    require 'optim'
    local _playerSimEvlLogger = optim.Logger(paths.concat('userModelTrained', 'userSimEvl', opt.uapFile..'_'..opt.uspFile..'_'..opt.ciRwdStMxTemp..'_'..'evl.log'))
    _playerSimEvlLogger:setNames{'Ite', 'Avg_adap', 'Adp_T1', 'Adp_T2', 'Adp_T3', 'Adp_T4', 'Avg_traj_len', 'Scr_p', 'Scr_n', 'Avg_len_p', 'Avg_len_n', 'Avg_scr'}
    local CIUserSimEnvModel = CIUserSimEnv(opt)

    local gens = opt.rlEvnIte
    local adpTotLen = 0
    local adpLenType = {0, 0, 0, 0}
    local totalTrajLength = 0
    local scoreStat = {0, 0}
    local totalLengthEachType = {0, 0}
    for i=1, gens do
        local obv, score, term, adpType
        local adpCnt = 0
        term = false
        obv, adpType = CIUserSimEnvModel:start()
        while not term do
            adpLenType[adpType] = adpLenType[adpType] + 1
            adpCnt = adpCnt + 1
            local rndAdpAct = torch.random(fr.ciAdpActRanges[adpType][1], fr.ciAdpActRanges[adpType][2])
            --            print('^--- Adaptation type', adpType, 'Random act choice: ', rndAdpAct)
            score, obv, term, adpType = CIUserSimEnvModel:step(rndAdpAct)
            --            print('^### Outside in main\n state:', obv, '\n type:', adpType, '\n score:', score, ',term:', term)
        end
        adpTotLen = adpTotLen + adpCnt
        totalTrajLength = totalTrajLength + CIUserSimEnvModel.timeStepCnt
        if score > 0.5 then
            scoreStat[1] = scoreStat[1] + 1
            totalLengthEachType[1] = totalLengthEachType[1] + CIUserSimEnvModel.timeStepCnt
        else
            scoreStat[2] = scoreStat[2] + 1
            totalLengthEachType[2] = totalLengthEachType[2] + CIUserSimEnvModel.timeStepCnt
        end
        if i % 50 == 0 then
            _playerSimEvlLogger:add{string.format('%d', i), string.format('%.3f', adpTotLen/i), string.format('%d', adpLenType[1]),
                string.format('%d', adpLenType[2]), string.format('%d', adpLenType[3]), string.format('%d', adpLenType[4]),
                string.format('%.3f', totalTrajLength/i), string.format('%d', scoreStat[1]), string.format('%d', scoreStat[2]),
                string.format('%.3f', totalLengthEachType[1]/scoreStat[1]), string.format('%.3f', totalLengthEachType[2]/scoreStat[2]),
                string.format('%.3f', (scoreStat[1] - scoreStat[2]) / i)}
            print('Avg_score: ', (scoreStat[1] - scoreStat[2]) / i)
        end
    end
    print('In user behaviro generation in', gens, 'times, avg adp appearances:', adpTotLen/gens, 'Adp types:', adpLenType)
    print('Avg user action traj length: ', totalTrajLength/gens, ', Score dist: ', scoreStat, 'Avg user act length of type 1: ',
        totalLengthEachType[1]/scoreStat[1], 'Avg user act length of type 2:', totalLengthEachType[2]/scoreStat[2])
elseif opt.trType == 'ev' and opt.uSimShLayer < 0.5 then
    local CIUserActsPred = CIUserActsPredictor(CIUserModel, opt)
    local CIUserScorePred = CIUserScorePredictor(CIUserModel, opt)
    CIUserActsPred.model:evaluate()
    CIUserScorePred.model:evaluate()
    local CIUserBehaviorGen = CIUserBehaviorGenEvaluator(CIUserModel, CIUserActsPred, CIUserScorePred, nil, opt)
elseif opt.trType == 'ev' and opt.uSimShLayer > 0.5 then
    local CIUserActScorePred = CIUserActScorePredictor(CIUserModel, opt)
    CIUserActScorePred.model:evaluate()
    local CIUserBehaviorGen = CIUserBehaviorGenEvaluator(CIUserModel, nil, nil, CIUserActScorePred, opt)
end



--print('@@', fr.traceData['100-0028'])
--print('#', #fr.data)
--print('#', #fr.data, ',', fr.data[1], '@@', fr.data[55][1], fr.data[55][81])