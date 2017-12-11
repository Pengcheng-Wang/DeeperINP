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
       -m,--uppModel      (default "rnn_lstm")  type of model to train: moe | mlp | linear | rnn_lstm | rnn_rhn | rnn_blstm | rnn_bGridlstm | cnn_uSimTempCnn
       --uppModelRNNDom   (default 0)           indicator of whether the model is an RNN model and uses dropout masks from outside of the model. 0 for not using outside mask. Otherwise, this number represents the number of gates used in RNN model
       --gridLstmTieWhts  (default 1)           indicator of whether the GridLSTM will have shared, tied weights along depth dimension. 1 means with shared weights, 0 means non-shared weights
       -f,--full                                use the full dataset
       -p,--plot                                plot while training
       -o,--optimization  (default "adam")       optimization: SGD | LBFGS | adam | rmsprop
       -r,--learningRate  (default 2e-4)        learning rate, for SGD only
       -b,--batchSize     (default 30)          batch size
       -m,--momentum      (default 0)           momentum, for SGD only
       -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
       --coefL1           (default 0)           L1 penalty on the weights
       --coefL2           (default 0)           L2 penalty on the weights
       -t,--threads       (default 4)           number of threads
       -g,--gpu           (default 0)           gpu device id, 0 for using cpu
       --seed             (default 1)           Random seed
       --prepro           (default "std")       input state feature preprocessing: rsc | std
       --rnnHdSizeL1      (default 32)          rnn hidden layer size
       --rnnHdSizeL2      (default 0)           rnn hidden layer size in 2nd lstm layer
       --rnnHdLyCnt       (default 2)           number of rnn/cnn hidden layer. Default is 2 bcz only when rnnHdSizeL2 is not 0 this opt will be examined. The RHN and Bayesian LSTM rnn number also uses this opt param. I'm also trying to use it for CNN hidden layer counting.
       --rhnReccDept      (default 5)           The recurrent depth of RHN model in one layer
       --lstmHist         (default 10)          lstm hist length. This influence the rnn tensor table construction in data preparation. Attention: we also use it as history length indicator (input frame number) in CNN models
       --cnnKernelWidth   (default 3)           Temporal Convolution kernel width
       --cnnConnType      (default "v2")        Residual connection type in player simulation CNN model
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
       --uSimScSoft       (default 0)           Whether to use soft labels in player score prediction. 1 for using soft labels, 0 for not.
       --rlEvnIte         (default 10000)       No of iterations in rl type of evaluation
       --usimTrIte        (default 400)         No of iterations used in user simulation model training. Recom for act training is 300, score training is 3000
       --termActSmgLen    (default 50)          The length above which user termination action would be highly probably sampled. The observed avg length is about 40
       --termActSmgEps    (default 0.9)         The probability which user termination action would be sampled after certain length
       --testSetDivSeed   (default 2)           The default seed value when separating a test set from the dataset
       --validSetDivSeed  (default 3)           The default seed value when separating a validation set out from the training set
       --dropoutUSim      (default 0.1)         The dropout rate used in user simulation model building. Set 0 to turn off droput
       --testOnTestFreq   (default 1)           The frequency of testing user simulation model's performance on test/train_valid set
       --scorePredStateScope     (default 60)   The range of distance of a player state to the end of the player interaction trajectory (ending state) that makes the state a valid training data poing for score prediction. The idea is to throw out early interaction state for user outcome predictor training
       --actPredDataAug   (default 1)           Whether to use data augmentation in action prediction model training. 0 for not using, 1 for using.
    ]]

-- threads and default tensor type
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.DoubleTensor')    --FloatTensor
-- Set manual seed
torch.manualSeed(opt.seed)

-- set the uppModelRNNDom indicator in opt, which indicates whether the model is an RNN model, and uses dropout mask from outside the model construction
-- right now, the rhn model, and Bayesian lstm model (following Gal's implementation), and GridLSTM model use outside dropout mask
if opt.uppModel == 'rnn_rhn' then
    -- rnn_rhn uses double-sized dropout mask to drop out inputs of calculation of t-gate and transformed inner cell state
    opt.uppModelRNNDom = 2
elseif opt.uppModel == 'rnn_blstm' or opt.uppModel == 'rnn_bGridlstm' then
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

if opt.trType == 'sc' and opt.uSimShLayer < 1 and opt.uSimScSoft < 1 then
    local CIUserScorePred = CIUserScorePredictor(CIUserModel, opt)
    for i=1, opt.usimTrIte do
        CIUserScorePred:trainOneEpoch()
    end
elseif opt.trType == 'sc' and opt.uSimShLayer < 1 and opt.uSimScSoft == 1 then
    local CIUserScoreSoftPredictor = require 'UserSimLearner/UserScoreSoftPredictor'
    local CIUserScoreSoftPred = CIUserScoreSoftPredictor(CIUserModel, opt)
    for i=1, opt.usimTrIte do
        CIUserScoreSoftPred:trainOneEpoch()
    end
elseif opt.trType == 'ac' and opt.uSimShLayer < 1 then
    local CIUserActsPred = CIUserActsPredictor(CIUserModel, opt)
    for i=1, opt.usimTrIte do
        CIUserActsPred:trainOneEpoch()
    end
elseif (opt.trType == 'ac' or opt.trType == 'sc') and opt.uSimShLayer == 1 then
    local CIUserActScorePred = CIUserActScorePredictor(CIUserModel, opt)
    for i=1, opt.usimTrIte do
        CIUserActScorePred:trainOneEpoch()
    end
elseif opt.trType == 'rl' then

    local CIUserSimEnvModel = CIUserSimEnv(opt)

    local gens = opt.rlEvnIte
    local adpTotLen = 0
    local adpLenType = {0, 0, 0, 0}
    local totalTrajLength = 0
    local scoreStat = {0, 0}
    local totalLengthEachType = {0, 0}
    for i=1, gens do
        print('iter', i)
        local obv, score, term, adpType
        local adpCnt = 0
        term = false
        obv, adpType = CIUserSimEnvModel:start()
        print('^### Outside in main\n state:', obv, '\n type:', adpType)
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
        if score > 0 then
            scoreStat[1] = scoreStat[1] + 1
            totalLengthEachType[1] = totalLengthEachType[1] + CIUserSimEnvModel.timeStepCnt
        else
            scoreStat[2] = scoreStat[2] + 1
            totalLengthEachType[2] = totalLengthEachType[2] + CIUserSimEnvModel.timeStepCnt
        end
    end
    print('In user behaviro generation in', gens, 'times, avg adp appearances:', adpTotLen/gens, 'Adp types:', adpLenType)
    print('Avg user action traj length: ', totalTrajLength/gens, ', Score dist: ', scoreStat, 'Avg user act length of type 1: ',
        totalLengthEachType[1]/scoreStat[1], 'Avg user act length of type 2:', totalLengthEachType[2]/scoreStat[2])
elseif opt.trType == 'ev' and opt.uSimShLayer < 1 then
    local CIUserActsPred = CIUserActsPredictor(CIUserModel, opt)
    local CIUserScorePred = CIUserScorePredictor(CIUserModel, opt)
    CIUserActsPred.model:evaluate()
    CIUserScorePred.model:evaluate()
    local CIUserBehaviorGen = CIUserBehaviorGenEvaluator(CIUserModel, CIUserActsPred, CIUserScorePred, nil, opt)
elseif opt.trType == 'ev' and opt.uSimShLayer == 1 then
    local CIUserActScorePred = CIUserActScorePredictor(CIUserModel, opt)
    CIUserActScorePred.model:evaluate()
    local CIUserBehaviorGen = CIUserBehaviorGenEvaluator(CIUserModel, nil, nil, CIUserActScorePred, opt)
end



--print('@@', fr.traceData['100-0028'])
--print('#', #fr.data)
--print('#', #fr.data, ',', fr.data[1], '@@', fr.data[55][1], fr.data[55][81])