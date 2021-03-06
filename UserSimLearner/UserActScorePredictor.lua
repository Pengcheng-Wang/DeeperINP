--
-- Created by IntelliJ IDEA.
-- User: pwang8
-- Date: 5/6/17
-- Time: 5:23 PM
-- This script is modified from UserActsPredictor.lua and UserScorePredictor.lua
-- This script creates a NN model which combines user action and score prediction.
--

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'rnn'
local nninit = require 'nninit'
local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'
local OptimMisc = require 'MyMisc.OptimMisc'    -- required to do gradient clipping for rnn modeling training

local CIUserActScorePredictor = classic.class('UserActScorePredictor')

function CIUserActScorePredictor:_init(CIUserSimulator, opt)

    if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
        error('LBFGS should not be used with small mini-batches; 1000 is recommended')
    end

    self.ciUserSimulator = CIUserSimulator
    self.opt = opt
    ----------------------------------------------------------------------
    -- define model to train
    -- on the 15-class user action classification problem
    -- and 2-class user outcome(score) classification problem
    --
    classesActs = {}
    classesScores = {}
    for i=1, CIUserSimulator.CIFr.usrActInd_end do classesActs[i] = i end   -- set action classes
    for i=1,2 do classesScores[i] = i end   -- set score(outcome) classes. When NLG is the metric, label 1 means pos NLG, label 2 means neg NLG
    self.inputFeatureNum = CIUserSimulator.realUserDataStates[1]:size()[1]  -- should be 18+3 now

    if opt.ciunet == '' then    -- ciunet is the CIUserActScorePredictor model to be loaded from file
        -- define model to train
        self.model = nn.Sequential()

        if opt.uppModel == 'moe' then
            ------------------------------------------------------------
            -- mixture of experts
            ------------------------------------------------------------
            local experts = nn.ConcatTable()
            local numOfExp = 4
            for i = 1, numOfExp do
                local expert = nn.Sequential()
                expert:add(nn.Reshape(self.inputFeatureNum))
                expert:add(nn.Linear(self.inputFeatureNum, 32))
                expert:add(nn.ReLU())
                if opt.dropoutUSim > 0 then expert:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any
                expert:add(nn.Linear(32, 24))
                expert:add(nn.ReLU())
                if opt.dropoutUSim > 0 then expert:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any

                -- The following code creates two output modules, with one module matches
                -- to user action prediction, and the other matches to user outcome(score) prediction
                local mulOutConcatTab = nn.ConcatTable()
                local actSeqNN = nn.Sequential()
                actSeqNN:add(nn.Linear(24, #classesActs))
                actSeqNN:add(nn.LogSoftMax())
                local scoreSeqNN = nn.Sequential()
                scoreSeqNN:add(nn.Linear(24, #classesScores))
                scoreSeqNN:add(nn.LogSoftMax())
                mulOutConcatTab:add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
                mulOutConcatTab:add(scoreSeqNN) -- {act, outcome(score)}

                expert:add(mulOutConcatTab)
                expert:add(nn.JoinTable(-1))
                experts:add(expert)
            end

            local gater = nn.Sequential()
            gater:add(nn.Reshape(self.inputFeatureNum))
            gater:add(nn.Linear(self.inputFeatureNum, 24))
            gater:add(nn.Tanh())
            if opt.dropoutUSim > 0 then gater:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any
            gater:add(nn.Linear(24, numOfExp))
            gater:add(nn.SoftMax())

            local trunk = nn.ConcatTable()
            trunk:add(gater)
            trunk:add(experts)

            self.model:add(trunk)
            self.model:add(nn.MixtureTable())
            ------------------------------------------------------------

        elseif opt.uppModel == 'mlp' then
            ------------------------------------------------------------
            -- regular 2-layer MLP
            ------------------------------------------------------------
            self.model:add(nn.Reshape(self.inputFeatureNum))
            self.model:add(nn.Linear(self.inputFeatureNum, 32))
            self.model:add(nn.ReLU())
            if opt.dropoutUSim > 0 then self.model:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any
            self.model:add(nn.Linear(32, 24))
            self.model:add(nn.ReLU())
            if opt.dropoutUSim > 0 then self.model:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any

            -- The following code creates two output modules, with one module matches
            -- to user action prediction, and the other matches to user outcome(score) prediction
            local mulOutConcatTab = nn.ConcatTable()
            local actSeqNN = nn.Sequential()
            actSeqNN:add(nn.Linear(24, #classesActs))
            actSeqNN:add(nn.LogSoftMax())
            local scoreSeqNN = nn.Sequential()
            scoreSeqNN:add(nn.Linear(24, #classesScores))
            scoreSeqNN:add(nn.LogSoftMax())
            mulOutConcatTab:add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
            mulOutConcatTab:add(scoreSeqNN) -- {act, outcome(score)}

            self.model:add(mulOutConcatTab)
            ------------------------------------------------------------

        elseif opt.uppModel == 'linear' then
            ------------------------------------------------------------
            -- simple linear model: logistic regression
            ------------------------------------------------------------
            -- Attention: this implementation with ConcatTable does not have
            -- any differences from separate act/score prediction implementation.
            -- Because the two modules have no shared params at all.
            -- So, should not try to use it.
            self.model:add(nn.Reshape(self.inputFeatureNum))

            -- The following code creates two output modules, with one module matches
            -- to user action prediction, and the other matches to user outcome(score) prediction
            local mulOutConcatTab = nn.ConcatTable()
            local actSeqNN = nn.Sequential()
            actSeqNN:add(nn.Linear(self.inputFeatureNum, #classesActs))
            actSeqNN:add(nn.LogSoftMax())
            local scoreSeqNN = nn.Sequential()
            scoreSeqNN:add(nn.Linear(self.inputFeatureNum, #classesScores))
            scoreSeqNN:add(nn.LogSoftMax())
            mulOutConcatTab:add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
            mulOutConcatTab:add(scoreSeqNN) -- {act, outcome(score)}

            self.model:add(mulOutConcatTab)
            ------------------------------------------------------------

        elseif opt.uppModel == 'rnn_lstm' then
            ------------------------------------------------------------
            -- lstm implementation from Element-Research rnn lib. The lazy dropout (variational RNN models) seems not very correct.
            ------------------------------------------------------------
            self.model:add(nn.Reshape(self.inputFeatureNum))
            --nn.FastLSTM.bn = true   -- turn on batch normalization
            local lstm
            if opt.uSimGru == 0 then
                lstm = nn.FastLSTM(self.inputFeatureNum, opt.rnnHdSizeL1, opt.uSimLstmBackLen, nil, nil, nil, opt.dropoutUSim) -- the 3rd param, [rho], the maximum amount of backpropagation steps to take back in time, default value is 9999
                TableSet.fastLSTMForgetGateInit(lstm, opt.dropoutUSim, opt.rnnHdSizeL1, nninit)
                -- has not applied batch normalization for fastLSTM before, should try it.
            else
                lstm = nn.GRU(self.inputFeatureNum, opt.rnnHdSizeL1, opt.uSimLstmBackLen, opt.dropoutUSim)   -- did not apply dropout or batchNormalization for GRU before
            end
            lstm:remember('both')
            self.model:add(lstm)
            self.model:add(nn.NormStabilizer())
            -- if need a 2nd lstm layer
            if opt.rnnHdSizeL2 ~= 0 then
                local lstmL2
                if opt.uSimGru == 0 then
                    lstmL2 = nn.FastLSTM(opt.rnnHdSizeL1, opt.rnnHdSizeL2, opt.uSimLstmBackLen, nil, nil, nil, opt.dropoutUSim) -- the 3rd param, [rho], the maximum amount of backpropagation steps to take back in time, default value is 9999
                    TableSet.fastLSTMForgetGateInit(lstmL2, opt.dropoutUSim, opt.rnnHdSizeL2, nninit)
                    -- has not applied batch normalization for fastLSTM before, should try it.
                else
                    lstmL2 = nn.GRU(opt.rnnHdSizeL1, opt.rnnHdSizeL2, opt.uSimLstmBackLen, opt.dropoutUSim)     -- did not apply dropout or batchNormalization for GRU before
                end
                lstmL2:remember('both')
                self.model:add(lstmL2)
                self.model:add(nn.NormStabilizer()) -- I am not very clear if NormStabilizer should be used togeher with Dropout, especially since dropout is used on memory value, equals to change memory value distribution a little
                -- If extra layers were needed. Right now we use the same setting for lstm layers that higher than 2
                for _extL=3, opt.rnnHdLyCnt do
                    local _extLstmL
                    if opt.uSimGru == 0 then
                        _extLstmL = nn.FastLSTM(opt.rnnHdSizeL2, opt.rnnHdSizeL2, opt.uSimLstmBackLen, nil, nil, nil, opt.dropoutUSim) -- the 3rd param, [rho], the maximum amount of backpropagation steps to take back in time, default value is 9999
                        TableSet.fastLSTMForgetGateInit(_extLstmL, opt.dropoutUSim, opt.rnnHdSizeL2, nninit)
                    else
                        _extLstmL = nn.GRU(opt.rnnHdSizeL2, opt.rnnHdSizeL2, opt.uSimLstmBackLen, opt.dropoutUSim)
                    end
                    _extLstmL:remember('both')
                    self.model:add(_extLstmL)
                    self.model:add(nn.NormStabilizer())
                end
            end
            local lastHidNum
            if opt.rnnHdSizeL2 == 0 then
                lastHidNum = opt.rnnHdSizeL1
            else
                lastHidNum = opt.rnnHdSizeL2
            end

            -- The following code creates two output modules, with one module matches
            -- to user action prediction, and the other matches to user outcome(score) prediction
            local mulOutConcatTab = nn.ConcatTable()
            local actSeqNN = nn.Sequential()
            actSeqNN:add(nn.Linear(lastHidNum, #classesActs))
            actSeqNN:add(nn.LogSoftMax())
            local scoreSeqNN = nn.Sequential()
            scoreSeqNN:add(nn.Linear(lastHidNum, #classesScores))
            scoreSeqNN:add(nn.LogSoftMax())
            mulOutConcatTab:add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
            mulOutConcatTab:add(scoreSeqNN) -- {act, outcome(score)}

            self.model:add(mulOutConcatTab)
            self.model = nn.Sequencer(self.model)
            ------------------------------------------------------------

        elseif opt.uppModel == 'rnn_rhn' then
            ------------------------------------------------------------
            -- Recurrent Highway Network (dropout mask defined outside rnn model)
            ------------------------------------------------------------
            require 'modules.RecurrenHighwayNetworkRNN'
            local rhn
            rhn = nn.RHN(self.inputFeatureNum, opt.rnnHdSizeL1, opt.rhnReccDept, opt.rnnHdLyCnt, opt.uSimLstmBackLen, opt.rnnResidual) --inputSize, outputSize, recurrence_depth, rhn_layers, rho, rnnResidual
            rhn:remember('both')
            self.model:add(rhn)
            self.model:add(nn.NormStabilizer())

            -- The following code creates two output modules, with one module matches
            -- to user action prediction, and the other matches to user outcome(score) prediction
            local mulOutConcatTab = nn.ConcatTable()
            local actSeqNN = nn.Sequential()
            actSeqNN:add(nn.Linear(opt.rnnHdSizeL1, #classesActs))
            actSeqNN:add(nn.LogSoftMax())
            local scoreSeqNN = nn.Sequential()
            scoreSeqNN:add(nn.Linear(opt.rnnHdSizeL1, #classesScores))
            scoreSeqNN:add(nn.LogSoftMax())
            mulOutConcatTab:add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
            mulOutConcatTab:add(scoreSeqNN) -- {act, outcome(score)}

            self.model:add(mulOutConcatTab)
            self.model = nn.Sequencer(self.model)
            ------------------------------------------------------------

        elseif opt.uppModel == 'rnn_blstm' then
            ------------------------------------------------------------
            -- Bayesian LSTM implemented following Yarin Gal's code (dropout mask defined outside rnn model)
            ------------------------------------------------------------
            require 'modules.LSTMBayesianRNN'
            local bay_lstm
            bay_lstm = nn.BayesianLSTM(self.inputFeatureNum, opt.rnnHdSizeL1, opt.rnnHdLyCnt, opt.uSimLstmBackLen) --inputSize, outputSize, rhn_layers, rho
            bay_lstm:remember('both')
            self.model:add(bay_lstm)
            self.model:add(nn.NormStabilizer())

            -- The following code creates two output modules, with one module matches
            -- to user action prediction, and the other matches to user outcome(score) prediction
            local mulOutConcatTab = nn.ConcatTable()
            local actSeqNN = nn.Sequential()
            actSeqNN:add(nn.Linear(opt.rnnHdSizeL1, #classesActs))
            actSeqNN:add(nn.LogSoftMax())
            local scoreSeqNN = nn.Sequential()
            scoreSeqNN:add(nn.Linear(opt.rnnHdSizeL1, #classesScores))
            scoreSeqNN:add(nn.LogSoftMax())
            mulOutConcatTab:add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
            mulOutConcatTab:add(scoreSeqNN) -- {act, outcome(score)}

            self.model:add(mulOutConcatTab)
            self.model = nn.Sequencer(self.model)
            ------------------------------------------------------------

        elseif opt.uppModel == 'rnn_bGridlstm' then
            ------------------------------------------------------------
            -- Bayesian GridLSTM implemented following Corey's GridLSTM and Yarin Gal's Bayesian LSTM code (dropout mask defined outside rnn model)
            ------------------------------------------------------------
            require 'modules.GridLSTMBayesianRNN'
            local grid_lstm
            grid_lstm = nn.BayesianGridLSTM(self.inputFeatureNum, opt.rnnHdLyCnt, opt.uSimLstmBackLen, opt.gridLstmTieWhts) -- rnn_size, rnn_layers, rho, tie_weights
            grid_lstm:remember('both')
            self.model:add(grid_lstm)
            self.model:add(nn.NormStabilizer())

            -- The following code creates two output modules, with one module matches
            -- to user action prediction, and the other matches to user outcome(score) prediction
            local mulOutConcatTab = nn.ConcatTable()
            local actSeqNN = nn.Sequential()
            actSeqNN:add(nn.Linear(self.inputFeatureNum, #classesActs))
            actSeqNN:add(nn.LogSoftMax())
            local scoreSeqNN = nn.Sequential()
            scoreSeqNN:add(nn.Linear(self.inputFeatureNum, #classesScores))
            scoreSeqNN:add(nn.LogSoftMax())
            mulOutConcatTab:add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
            mulOutConcatTab:add(scoreSeqNN) -- {act, outcome(score)}

            self.model:add(mulOutConcatTab)
            self.model = nn.Sequencer(self.model)
            ------------------------------------------------------------

        elseif opt.uppModel == 'cnn_uSimTempCnn' then
            ------------------------------------------------------------
            -- CNN model following the implementation of OpenNMT CNNEncoder and fb.resnet
            ------------------------------------------------------------
            require 'modules.TempConvInUserSimCNN'
            local tempCnn = nn.TempConvUserSimCNN()         -- inputSize, outputSize, cnn_layers, kernel_width, dropout_rate, version
            local _tempCnnLayer = tempCnn:CreateCNNModule(self.inputFeatureNum, self.inputFeatureNum, opt.rnnHdLyCnt, opt.cnnKernelWidth, opt.dropoutUSim, opt.lstmHist, opt.cnnConnType)
            self.model:add(_tempCnnLayer)
            self.model:add(nn.View(-1):setNumInputDims(2))  -- The input/output data should have dimensions of batch_index/frame_index/feature_index, so it's 3d, and 2d without batch index

            -- The following code creates two output modules, with one module matches
            -- to user action prediction, and the other matches to user outcome(score) prediction
            local mulOutConcatTab = nn.ConcatTable()
            local actSeqNN = nn.Sequential()
            actSeqNN:add(nn.Linear(self.inputFeatureNum * opt.lstmHist, #classesActs))
            actSeqNN:add(nn.LogSoftMax())
            local scoreSeqNN = nn.Sequential()
            scoreSeqNN:add(nn.Linear(self.inputFeatureNum * opt.lstmHist, #classesScores))
            scoreSeqNN:add(nn.LogSoftMax())
            mulOutConcatTab:add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
            mulOutConcatTab:add(scoreSeqNN) -- {act, outcome(score)}

            self.model:add(mulOutConcatTab)
            ------------------------------------------------------------

        else
            print('Unknown uppModel type'..opt.uppModel..' in UserActScorePredictor training')
            torch.CmdLine():text()
            error()
        end

        -- params init
        local uapLinearLayers = self.model:findModules('nn.Linear')
        for l = 1, #uapLinearLayers do
            uapLinearLayers[l]:init('weight', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)}):init('bias', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)})    -- This bias initialization seems a little bit conflict with FastLSTM forget gate bias init. Maybe not, bcz it's on linear modules.
        end
    elseif opt.ciunet == 'rlLoad' then  -- If need reload a trained uasp model in the RL training/evaluation, not for training uasp anymore
        if string.sub(opt.uppModel, 1, 7) == 'rnn_rhn' then
            require 'modules.RecurrenHighwayNetworkRNN'
        elseif opt.uppModel == 'rnn_blstm' then
            require 'modules.LSTMBayesianRNN'
        elseif opt.uppModel == 'rnn_bGridlstm' then
            require 'modules.GridLSTMBayesianRNN'
        elseif string.sub(opt.uppModel, 1, 4) == 'cnn_' then
            require 'modules.TempConvInUserSimCNN'
        end

        self.model = torch.load(paths.concat(opt.ubgDir , opt.uapFile))
    else
        print('<trainer> reloading previously trained ciunet')
        self.model = torch.load(opt.ciunet)
    end

    --    -- verbose
    --    print(self.model)

    ----------------------------------------------------------------------
    -- loss function: negative log-likelihood
    --
    self.uaspPrlCriterion = nn.ParallelCriterion()
    self.uapCriterion = nn.ClassNLLCriterion()
    self.uspCriterion = nn.ClassNLLCriterion()
    self.uaspPrlCriterion:add(self.uapCriterion)   -- action prediction loss function
    self.uaspPrlCriterion:add(self.uspCriterion)   -- score (outcome) prediction loss function
    if string.sub(opt.uppModel, 1, 4) == 'rnn_' then
        self.uaspPrlCriterion = nn.SequencerCriterion(self.uaspPrlCriterion)
    end

    self.trainEpoch = 1
    -- these matrices records the current confusion across classesActs and classesScores
    self.uapConfusion = optim.ConfusionMatrix(classesActs)
    self.uspConfusion = optim.ConfusionMatrix(classesScores)

    -- log results to files
    self.uaspTrainLogger = optim.Logger(paths.concat('userModelTrained', opt.save, 'uaspTrain.log'))
    self.uaspTestLogger = optim.Logger(paths.concat('userModelTrained', opt.save, 'uaspTest.log'))
    self.uaspTestLogger:setNames{'Epoch', 'Act Test acc.', 'Act Test LogLoss', 'Score Test acc.', 'Score Test LogLoss'}

    ----------------------------------------------------------------------
    --- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
    ---
    if opt.gpu > 0 then
        local ok, cunn = pcall(require, 'cunn')
        local ok2, cutorch = pcall(require, 'cutorch')
        if not ok then print('package cunn not found!') end
        if not ok2 then print('package cutorch not found!') end
        if ok and ok2 then
            print('using CUDA on GPU ' .. opt.gpu .. '...')
            cutorch.setDevice(opt.gpu)
            cutorch.manualSeed(opt.seed)
            --- set up cuda nn
            self.model = self.model:cuda()
            self.uaspPrlCriterion = self.uaspPrlCriterion:cuda()
        else
            print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
            print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
            print('Falling back on CPU mode')
            opt.gpu = 0 -- overwrite user setting
        end
    end

    ----------------------------------------------------------------------
    --- Prepare data for RNN models in training set
    ---
    self.rnnRealUserDataStates = self.ciUserSimulator.rnnRealUserDataStates
    self.rnnRealUserDataActs = self.ciUserSimulator.rnnRealUserDataActs
    self.rnnRealUserDataRewards = self.ciUserSimulator.rnnRealUserDataRewards
    self.rnnRealUserDataStandardNLG = self.ciUserSimulator.rnnRealUserDataStandardNLG
    self.rnnRealUserDataEnds = self.ciUserSimulator.rnnRealUserDataEnds

    ----------------------------------------------------------------------
    --- Prepare data for RNN models in test/train_validation set
    ---
    self.rnnRealUserDataStatesTest = self.ciUserSimulator.rnnRealUserDataStatesTest
    self.rnnRealUserDataActsTest = self.ciUserSimulator.rnnRealUserDataActsTest
    self.rnnRealUserDataRewardsTest = self.ciUserSimulator.rnnRealUserDataRewardsTest
    self.rnnRealUserDataEndsTest = self.ciUserSimulator.rnnRealUserDataEndsTest

    ----------------------------------------------------------------------
    --- Prepare data for CNN models in training set
    ---
    self.cnnRealUserDataStates = self.ciUserSimulator.cnnRealUserDataStates
    self.cnnRealUserDataActs = self.ciUserSimulator.cnnRealUserDataActs
    self.cnnRealUserDataRewards = self.ciUserSimulator.cnnRealUserDataRewards
    self.cnnRealUserDataStandardNLG = self.ciUserSimulator.cnnRealUserDataStandardNLG

    ----------------------------------------------------------------------
    --- Prepare data for CNN models in test/train_validation set
    ---
    self.cnnRealUserDataStatesTest = self.ciUserSimulator.cnnRealUserDataStatesTest
    self.cnnRealUserDataActsTest = self.ciUserSimulator.cnnRealUserDataActsTest
    self.cnnRealUserDataRewardsTest = self.ciUserSimulator.cnnRealUserDataRewardsTest
    self.cnnRealUserDataEndsTest = self.ciUserSimulator.cnnRealUserDataEndsTest

    ----------------------------------------------------------------------
    --- Prepare 3 dropout masks for RNN models. Right now
    --- it is used by RHN, Bayesian LSTM and Bayesian GridLSTM
    ---
    self.rnn_noise_i = {}
    self.rnn_noise_h = {}
    self.rnn_noise_o = {}
    if self.opt.uppModelRNNDom > 0 then
        TableSet.buildRNNDropoutMask(self.rnn_noise_i, self.rnn_noise_h, self.rnn_noise_o, self.inputFeatureNum, opt.rnnHdSizeL1, opt.rnnHdLyCnt, self.opt.batchSize, self.opt.lstmHist, self.opt.uppModelRNNDom)
    end

    -- retrieve parameters and gradients
    -- have to put these lines here below the gpu setting
    self.uaspParam, self.uaspDParam = self.model:getParameters()
end


-- training function
function CIUserActScorePredictor:trainOneEpoch()
    -- local vars
    local time = sys.clock()

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. self.trainEpoch .. ' [batchSize = ' .. self.opt.batchSize .. ']')
    local inputs
    local targetsActScore = {}
    local targetsAct
    local targetsScore
    local score_reg -- the regression target for score/outcome prediction output, in this script this value will only be used in data augmentation
    local closeToEnd
    local t = 1
    local lstmIter = 1  -- lstm iterate for each squence starts from this value
    local epochDone = false
    while not epochDone do
        if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
            -- rnn models
            inputs = {}
            targetsAct = {}
            targetsScore = {}
            score_reg = {}
            closeToEnd = torch.Tensor(self.opt.batchSize):fill(0)
            local k
            for j = 1, self.opt.lstmHist do
                inputs[j] = torch.Tensor(self.opt.batchSize, self.inputFeatureNum)
                targetsAct[j] = torch.Tensor(self.opt.batchSize)
                targetsScore[j] = torch.Tensor(self.opt.batchSize)
                score_reg[j] = torch.Tensor(self.opt.batchSize)
                k = 1
                for i = lstmIter, math.min(lstmIter+self.opt.batchSize-1, #self.rnnRealUserDataStates) do
                    inputs[j][k] = self.rnnRealUserDataStates[i][j]
                    targetsAct[j][k] = self.rnnRealUserDataActs[i][j]
                    targetsScore[j][k] = self.rnnRealUserDataRewards[i][j]
                    score_reg[j][k] = self.rnnRealUserDataStandardNLG[i][j] -- the score/outcome regression prediction ground-truth
                    if j == self.opt.lstmHist then
                        for dis=0, self.opt.scorePredStateScope-1 do
                            if (i+dis) <= #self.rnnRealUserDataActs and self.rnnRealUserDataActs[i+dis][self.opt.lstmHist] == self.ciUserSimulator.CIFr.usrActInd_end then
                                -- If current state is close enough to the end of this sequence, mark it.
                                -- This is for marking near end state, with which the score prediction should be more accurate and be utilized in score pred training
                                closeToEnd[k] = 1
                                break
                            end
                        end
                    end
                    k = k + 1
                end
            end

            -- at the end of dataset, if it could not be divided into full batch
            if k ~= self.opt.batchSize + 1 then
                while k <= self.opt.batchSize do
                    local randInd = torch.random(1, #self.rnnRealUserDataStates)
                    for j = 1, self.opt.lstmHist do
                        inputs[j][k] = self.rnnRealUserDataStates[randInd][j]
                        targetsAct[j][k] = self.rnnRealUserDataActs[randInd][j]
                        targetsScore[j][k] = self.rnnRealUserDataRewards[randInd][j]
                        score_reg[j][k] = self.rnnRealUserDataStandardNLG[randInd][j]
                        if j == self.opt.lstmHist then
                            for dis=0, self.opt.scorePredStateScope-1 do
                                if (randInd+dis) <= #self.rnnRealUserDataActs and self.rnnRealUserDataActs[randInd+dis][self.opt.lstmHist] == self.ciUserSimulator.CIFr.usrActInd_end then
                                    -- If current state is close enough to the end of this sequence, mark it.
                                    -- This is for marking near end state, with which the score prediction should be more accurate and be utilized in score pred training
                                    closeToEnd[k] = 1
                                    break
                                end
                            end
                        end
                    end
                    k = k + 1
                end
            end

            lstmIter = lstmIter + self.opt.batchSize
            if lstmIter > #self.rnnRealUserDataStates then
                epochDone = true
            end

            if self.opt.actPredDataAug > 0 then
                -- Data augmentation
                self.ciUserSimulator:UserSimActDataAugment(inputs, targetsAct, targetsScore, score_reg, self.opt.uppModel)
                if self.opt.uppModelRNNDom > 0 then
                    TableSet.buildRNNDropoutMask(self.rnn_noise_i, self.rnn_noise_h, self.rnn_noise_o, self.inputFeatureNum, self.opt.rnnHdSizeL1, self.opt.rnnHdLyCnt, inputs[1]:size(1), self.opt.lstmHist, self.opt.uppModelRNNDom)
                end
            end
            -- Should do input feature pre-processing after data augmentation
            for ik=1, #inputs do
                inputs[ik] = self.ciUserSimulator:preprocessUserStateData(inputs[ik], self.opt.prepro)
            end
            -- Try to add random normal noise to input features and see how it performs
            -- This should be invoked after input preprocess bcz we want to set an unique std
            -- I've tried to apply adding random normal noise in rnn form of data. It seems the result is not good.
            --self.ciUserSimulator:UserSimDataAddRandNoise(inputs, true, 0.01)

            if self.opt.uppModelRNNDom > 0 then
                TableSet.sampleRNNDropoutMask(self.opt.dropoutUSim, self.rnn_noise_i, self.rnn_noise_h, self.rnn_noise_o, self.opt.rnnHdLyCnt, self.opt.lstmHist)
                for j = 1, self.opt.lstmHist do
                    inputs[j] = {inputs[j], self.rnn_noise_i[j], self.rnn_noise_h[j], self.rnn_noise_o[j]}
                end
            end

            if self.opt.gpu > 0 then
                nn.utils.recursiveType(inputs, 'torch.CudaTensor')
                nn.utils.recursiveType(targetsAct, 'torch.CudaTensor')
                nn.utils.recursiveType(targetsScore, 'torch.CudaTensor')
                closeToEnd = closeToEnd:cuda()
                -- do not need transform score_reg because it is not used in training, only used in data augmentation
            end

            for j = 1, self.opt.lstmHist do
                targetsActScore[j] = {targetsAct[j], targetsScore[j]}
            end

        elseif string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then
            -- cnn models
            inputs = torch.Tensor(self.opt.batchSize, self.opt.lstmHist, self.inputFeatureNum)
            targetsAct = torch.Tensor(self.opt.batchSize)
            targetsScore = torch.Tensor(self.opt.batchSize)
            score_reg = torch.Tensor(self.opt.batchSize)  -- standard nlg for score regression
            closeToEnd = torch.Tensor(self.opt.batchSize):fill(0)
            local k = 1
            for i = t, math.min(t+self.opt.batchSize-1, #self.cnnRealUserDataStates) do
                inputs[k] = self.cnnRealUserDataStates[i]
                targetsAct[k] = self.cnnRealUserDataActs[i]
                targetsScore[k] = self.cnnRealUserDataRewards[i]
                score_reg[k] = self.cnnRealUserDataStandardNLG[i]
                for dis=0, self.opt.scorePredStateScope-1 do
                    if (i+dis) <= #self.cnnRealUserDataActs and self.cnnRealUserDataActs[i+dis] == self.ciUserSimulator.CIFr.usrActInd_end then
                        -- If current state is close enough to the end of this sequence, mark it.
                        -- This is for marking near end state, with which the score prediction should be more accurate and be utilized in score pred training
                        closeToEnd[k] = 1
                        break
                    end
                end
                k = k + 1
            end

            -- at the end of dataset, if it could not be divided into full batch
            if k ~= self.opt.batchSize + 1 then
                while k <= self.opt.batchSize do
                    local randInd = torch.random(1, #self.cnnRealUserDataStates)
                    inputs[k] = self.cnnRealUserDataStates[randInd]
                    targetsAct[k] = self.cnnRealUserDataActs[randInd]
                    targetsScore[k] = self.cnnRealUserDataRewards[randInd]
                    score_reg[k] = self.cnnRealUserDataStandardNLG[randInd]
                    for dis=0, self.opt.scorePredStateScope-1 do
                        if (randInd+dis) <= #self.cnnRealUserDataActs and self.cnnRealUserDataActs[randInd+dis] == self.ciUserSimulator.CIFr.usrActInd_end then
                            -- If current state is close enough to the end of this sequence, mark it.
                            -- This is for marking near end state, with which the score prediction should be more accurate and be utilized in score pred training
                            closeToEnd[k] = 1
                            break
                        end
                    end
                    k = k + 1
                end
            end

            t = t + self.opt.batchSize
            if t > #self.cnnRealUserDataStates then
                epochDone = true
            end

            if self.opt.actPredDataAug > 0 then
                -- Data augmentation
                self.ciUserSimulator:UserSimActDataAugment(inputs, targetsAct, targetsScore, score_reg, self.opt.uppModel)
            end
            -- Should do input feature pre-processing after data augmentation
            inputs = self.ciUserSimulator:preprocessUserStateData(inputs, self.opt.prepro)
            -- Try to add random normal noise to input features and see how it performs
            -- This should be invoked after input preprocess bcz we want to set an unique std
            --self.ciUserSimulator:UserSimDataAddRandNoise(inputs, false, 0.01)

            if self.opt.gpu > 0 then
                inputs = inputs:cuda()
                targetsAct = targetsAct:cuda()
                targetsScore = targetsScore:cuda()
                closeToEnd = closeToEnd:cuda()
            end

            targetsActScore = {targetsAct, targetsScore}

        else
            -- non-rnn, non-cnn models, create mini batch
            inputs = torch.Tensor(self.opt.batchSize, self.inputFeatureNum)
            targetsAct = torch.Tensor(self.opt.batchSize)
            targetsScore = torch.Tensor(self.opt.batchSize)
            score_reg = torch.Tensor(self.opt.batchSize)    -- score/outcome regression ground-truth
            closeToEnd = torch.Tensor(self.opt.batchSize):fill(0)
            local k = 1
            for i = t, math.min(t+self.opt.batchSize-1, #self.ciUserSimulator.realUserDataStates) do
                inputs[k] = self.ciUserSimulator.realUserDataStates[i]
                targetsAct[k] = self.ciUserSimulator.realUserDataActs[i]
                targetsScore[k] = self.ciUserSimulator.realUserDataRewards[i]
                score_reg[k] = self.ciUserSimulator.realUserDataStandardNLG[i]
                for dis=0, self.opt.scorePredStateScope-1 do
                    if (i+dis) <= #self.ciUserSimulator.realUserDataActs and self.ciUserSimulator.realUserDataActs[i+dis] == self.ciUserSimulator.CIFr.usrActInd_end then
                        -- If current state is close enough to the end of this sequence, mark it.
                        -- This is for marking near end state, with which the score prediction should be more accurate and be utilized in score pred training
                        closeToEnd[k] = 1
                        break
                    end
                end
                k = k + 1
            end

            -- at the end of dataset, if it could not be divided into full batch
            if k ~= self.opt.batchSize + 1 then
                while k <= self.opt.batchSize do
                    local randInd = torch.random(1, #self.ciUserSimulator.realUserDataStates)
                    inputs[k] = self.ciUserSimulator.realUserDataStates[randInd]
                    targetsAct[k] = self.ciUserSimulator.realUserDataActs[randInd]
                    targetsScore[k] = self.ciUserSimulator.realUserDataRewards[randInd]
                    score_reg[k] = self.ciUserSimulator.realUserDataStandardNLG[randInd]
                    for dis=0, self.opt.scorePredStateScope-1 do
                        if (randInd+dis) <= #self.ciUserSimulator.realUserDataActs and self.ciUserSimulator.realUserDataActs[randInd+dis] == self.ciUserSimulator.CIFr.usrActInd_end then
                            -- If current state is close enough to the end of this sequence, mark it.
                            -- This is for marking near end state, with which the score prediction should be more accurate and be utilized in score pred training
                            closeToEnd[k] = 1
                            break
                        end
                    end
                    k = k + 1
                end
            end

            t = t + self.opt.batchSize
            if t > #self.ciUserSimulator.realUserDataStates then
                epochDone = true
            end

            if self.opt.actPredDataAug > 0 then
                -- Data augmentation
                self.ciUserSimulator:UserSimActDataAugment(inputs, targetsAct, targetsScore, score_reg, self.opt.uppModel)
            end
            -- Should do input feature pre-processing after data augmentation
            inputs = self.ciUserSimulator:preprocessUserStateData(inputs, self.opt.prepro)
            -- Try to add random normal noise to input features and see how it performs
            -- This should be invoked after input preprocess bcz we want to set an unique std
            --self.ciUserSimulator:UserSimDataAddRandNoise(inputs, false, 0.01)

            if self.opt.gpu > 0 then
                inputs = inputs:cuda()
                targetsAct = targetsAct:cuda()
                targetsScore = targetsScore:cuda()
                closeToEnd = closeToEnd:cuda()
            end

            targetsActScore = {targetsAct, targetsScore}

        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- just in case:
            collectgarbage()

            -- get new parameters
            if x ~= self.uaspParam then
                self.uaspParam:copy(x)
            end

            -- reset gradients
            self.uaspDParam:zero()

            -- evaluate function for complete mini batch
            local outputs = self.model:forward(inputs)

            -- Here, if moe is used with shared lower layers, it is the problem that,
            -- due to limitation of MixtureTable module, we have to join tables together as
            -- a single tensor as output of the whole user action and score prediction model.
            -- So, to guarantee the compatability, we need split the tensor into two tables here,
            -- for act prediction and score prediction respectively.
            if self.opt.uppModel == 'moe' then
                outputs = outputs:split(#classesActs, 2)  -- We assume 1st dim is batch index. Act pred is the 1st set of output, having dim of 15. Score dim 2.
            end

            -- Zero error values (change output to target) for score prediction cases far away from ending state (I don't hope these cases influence training)
            for cl=1, closeToEnd:size(1) do
                if closeToEnd[cl] < 1 then
                    if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
                        outputs[self.opt.lstmHist][2][cl][targetsActScore[self.opt.lstmHist][2][cl]] = 0    -- set the ground-truth labeled item into 0
                    else
                        outputs[2][cl][targetsActScore[2][cl]] = 0
                    end
                end
            end

            local f = self.uaspPrlCriterion:forward(outputs, targetsActScore) -- I made an experiment. The returned error value (f) from parallelCriterion is the sum of each criterion
            local df_do = self.uaspPrlCriterion:backward(outputs, targetsActScore)

            if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
                local subTot_f = 0
                for s=1, self.opt.lstmHist do
                    subTot_f = subTot_f + self.uapCriterion:forward(outputs[s][1], targetsActScore[s][1])
                end
                f = subTot_f + self.uspCriterion:forward(outputs[self.opt.lstmHist][2], targetsActScore[self.opt.lstmHist][2])

                for step=1, self.opt.lstmHist-1 do
                    df_do[step][2]:zero()   -- Zero df_do over Score prediction from time 1 to listHist-1
                end
            end

            -- If moe with shared lower layers, we merge the df_do before backprop. This is necessary because the network has merged act/outcome prediction output in one tensor
            if self.opt.uppModel == 'moe' then
                df_do = nn.JoinTable(-1):forward(df_do)  -- We assume 1st dim is batch index. Act pred is the 1st set of output, having dim of 15. Score dim 2.
                if self.opt.gpu > 0 then
                    df_do = df_do:cuda()  -- seems like after calling the forward function of nn.JoinTable, that output(df_do) becomes main memory object again
                end
            end

            self.model:backward(inputs, df_do)

            -- penalties (L1 and L2):
            if self.opt.coefL1 ~= 0 or self.opt.coefL2 ~= 0 then
                -- locals:
                local norm,sign= torch.norm,torch.sign

                -- Loss:
                f = f + self.opt.coefL1 * norm(self.uaspParam,1)
                f = f + self.opt.coefL2 * norm(self.uaspParam,2)^2/2

                -- Gradients:
                self.uaspDParam:add( sign(self.uaspParam):mul(self.opt.coefL1) + self.uaspParam:clone():mul(self.opt.coefL2) )
            end

            -- update self.uapConfusion and self.uspConfusion
            if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
                for j = 1, self.opt.lstmHist do
                    for i = 1,self.opt.batchSize do
                        self.uapConfusion:add(outputs[j][1][i], targetsActScore[j][1][i])
                    end
                end
                for i = 1,self.opt.batchSize do
                    self.uspConfusion:add(outputs[self.opt.lstmHist][2][i], targetsActScore[self.opt.lstmHist][2][i])
                end
            else
                for i = 1,self.opt.batchSize do
                    self.uapConfusion:add(outputs[1][i], targetsActScore[1][i])
                    self.uspConfusion:add(outputs[2][i], targetsActScore[2][i])
                end
            end

            -- gradient clipping. It is recommended for rnn models, not sure if it is helpful to other models.
            if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
                OptimMisc.clipGradByNorm(self.uaspDParam, 10)    -- right now 10 is used constantly as clipping norm
            end
            -- return f and df/dX
            return f, self.uaspDParam
        end

        self.model:training()
        if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
            self.model:forget()
        end

        -- optimize on current mini-batch
        if self.opt.optimization == 'LBFGS' then

            -- Perform LBFGS step:
            lbfgsState = lbfgsState or {
                maxIter = self.opt.maxIter,
                lineSearch = optim.lswolfe
            }
            optim.lbfgs(feval, self.uaspParam, lbfgsState)

            -- disp report:
            print('LBFGS step')
            print(' - progress in batch: ' .. t .. '/' .. #self.ciUserSimulator.realUserDataStates)
            print(' - nb of iterations: ' .. lbfgsState.nIter)
            print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

        elseif self.opt.optimization == 'SGD' then

            -- Perform SGD step:
            sgdState = sgdState or {
                learningRate = self.opt.learningRate,
                momentum = self.opt.momentum,
                nesterov = true,
                dampening = 0,
                learningRateDecay = 5e-7
            }
            optim.sgd(feval, self.uaspParam, sgdState)

            -- disp progress
            if string.sub(self.opt.uppModel, 1, 4) ~= 'rnn_' then
                xlua.progress(t, #self.ciUserSimulator.realUserDataStates)
            else
                xlua.progress(lstmIter, #self.rnnRealUserDataStates)
            end


        elseif self.opt.optimization == 'adam' then

            -- Perform Adam step:
            adamState = adamState or {
                learningRate = self.opt.learningRate,
                learningRateDecay = 5e-7
            }
            optim.adam(feval, self.uaspParam, adamState)

            -- disp progress
            if string.sub(self.opt.uppModel, 1, 4) ~= 'rnn_' then
                xlua.progress(t, #self.ciUserSimulator.realUserDataStates)
            else
                xlua.progress(lstmIter, #self.rnnRealUserDataStates)
            end

        elseif self.opt.optimization == 'rmsprop' then

            -- Perform Adam step:
            rmspropState = rmspropState or {
                learningRate = self.opt.learningRate
            }
            optim.rmsprop(feval, self.uaspParam, rmspropState)

            -- disp progress
            if string.sub(self.opt.uppModel, 1, 4) ~= 'rnn_' then
                xlua.progress(t, #self.ciUserSimulator.realUserDataStates)
            else
                xlua.progress(lstmIter, #self.rnnRealUserDataStates)
            end

        else
            error('unknown optimization method')
        end
    end

    -- time taken
    time = sys.clock() - time
    --    time = time / #self.ciUserSimulator.realUserDataStates
    print("<trainer> time to learn 1 epoch = " .. (time*1000) .. 'ms')


    self.uapConfusion:updateValids()
    local confMtxStr = 'Act prediction: average row correct: ' .. (self.uapConfusion.averageValid*100) .. '% \n' ..
            'average rowUcol correct (VOC measure): ' .. (self.uapConfusion.averageUnionValid*100) .. '% \n' ..
            ' + global correct: ' .. (self.uapConfusion.totalValid*100) .. '%'
    print(confMtxStr)

    self.uspConfusion:updateValids()
    confMtxStr = 'Score prediction: average row correct: ' .. (self.uspConfusion.averageValid*100) .. '% \n' ..
            'average rowUcol correct (VOC measure): ' .. (self.uspConfusion.averageUnionValid*100) .. '% \n' ..
            ' + global correct: ' .. (self.uspConfusion.totalValid*100) .. '%'
    print(confMtxStr)
    self.uaspTrainLogger:add{['% Act Prediction: mean class accuracy (train set)'] = self.uapConfusion.totalValid * 100,
        ['% Score Prediction: mean class accuracy (train set)'] = self.uspConfusion.totalValid * 100}


    -- save/log current net
    local filename = paths.concat('userModelTrained', self.opt.save, 'uasp.t7')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    --    if paths.filep(filename) then
    --        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    --    end
    --print('<trainer> saving ciunet to '..filename)
    --torch.save(filename, self.model)

    if self.trainEpoch % 10 == 0 and self.opt.ciuTType == 'train' then
        -- todo:pwang8. Dec 8, 2017. For test purpose, this model saving func is temporarily ceased
        --filename = paths.concat('userModelTrained', self.opt.save, string.format('%d', self.trainEpoch)..'_'..
        --        string.format('%.2f', self.uapConfusion.totalValid*100)..'_'..string.format('%.2f', self.uspConfusion.totalValid*100)..'uasp.t7')
        --os.execute('mkdir -p ' .. sys.dirname(filename))
        --print('<trainer> saving periodly trained ciunet to '..filename)
        --torch.save(filename, self.model)
    end

    if (self.opt.ciuTType == 'train' or self.opt.ciuTType == 'train_tr') and self.trainEpoch % self.opt.testOnTestFreq == 0 then
        local actScoreTestAccu = self:testActScorePredOnTestDetOneEpoch()
        print('<Act prediction accuracy at epoch '..string.format('%d', self.trainEpoch)..' on test set > '..string.format('%.2f%%', actScoreTestAccu[1]*100)..
            ', and LogLoss '..string.format('%.2f', actScoreTestAccu[2]))
        print('<Score prediction accuracy at epoch '..string.format('%d', self.trainEpoch)..' on test set > '..string.format('%.2f%%', actScoreTestAccu[3]*100)..
            ', and LogLoss '..string.format('%.2f', actScoreTestAccu[4]))
        self.uaspTestLogger:add{string.format('%d', self.trainEpoch), string.format('%.5f%%', actScoreTestAccu[1]*100), string.format('%.5f%%', actScoreTestAccu[2]),
            string.format('%.5f%%', actScoreTestAccu[3]*100), string.format('%.5f%%', actScoreTestAccu[4])}
    end

    self.uapConfusion:zero()
    self.uspConfusion:zero()
    -- next epoch
    self.trainEpoch = self.trainEpoch + 1
end

-- evaluation functionon test/train_validation set
function CIUserActScorePredictor:testActScorePredOnTestDetOneEpoch()
    -- just in case:
    collectgarbage()

    local _logLoss_act = 0
    local _logLoss_score = 0
    if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
        -- uSimShLayer == 1, rnn model
        self.model:forget()
        self.model:evaluate()

        local tabState = {}
        for j=1, self.opt.lstmHist do
            local prepUserState = torch.Tensor(#self.rnnRealUserDataStatesTest, self.ciUserSimulator.userStateFeatureCnt)
            for k=1, #self.rnnRealUserDataStatesTest do
                prepUserState[k] = self.ciUserSimulator:preprocessUserStateData(self.rnnRealUserDataStatesTest[k][j], self.opt.prepro)
            end
            tabState[j] = prepUserState
        end

        local test_rnn_noise_i = {}
        local test_rnn_noise_h = {}
        local test_rnn_noise_o = {}
        if self.opt.uppModelRNNDom > 0 then
            TableSet.buildRNNDropoutMask(test_rnn_noise_i, test_rnn_noise_h, test_rnn_noise_o, self.inputFeatureNum, self.opt.rnnHdSizeL1, self.opt.rnnHdLyCnt, #self.rnnRealUserDataStatesTest, self.opt.lstmHist, self.opt.uppModelRNNDom)
            TableSet.sampleRNNDropoutMask(0, test_rnn_noise_i, test_rnn_noise_h, test_rnn_noise_o, self.opt.rnnHdLyCnt, self.opt.lstmHist)
            for j = 1, self.opt.lstmHist do
                tabState[j] = {tabState[j], test_rnn_noise_i[j], test_rnn_noise_h[j], test_rnn_noise_o[j]}
            end
        end

        if self.opt.gpu > 0 then
            nn.utils.recursiveType(tabState, 'torch.CudaTensor')
        end
        local nll_acts = self.model:forward(tabState)
        nn.utils.recursiveType(nll_acts, 'torch.FloatTensor')
        if nll_acts[self.opt.lstmHist][1]:ne(nll_acts[self.opt.lstmHist][1]):sum() > 0 then print('nan appears in output!') os.exit() end
        if nll_acts[self.opt.lstmHist][2]:ne(nll_acts[self.opt.lstmHist][2]):sum() > 0 then print('nan appears in output!') os.exit() end

        --- Action prediction evaluation
        self.uapConfusion:zero()
        for i=1, #self.rnnRealUserDataStatesTest do
            self.uapConfusion:add(nll_acts[self.opt.lstmHist][1][i], self.rnnRealUserDataActsTest[i][self.opt.lstmHist])
            _logLoss_act = _logLoss_act + -1 * nll_acts[self.opt.lstmHist][1][i][self.rnnRealUserDataActsTest[i][self.opt.lstmHist]]
        end
        self.uapConfusion:updateValids()
        local tvalidAct = self.uapConfusion.totalValid
        self.uapConfusion:zero()

        --- Score prediction evaluation
        self.uspConfusion:zero()
        for i=1, #self.rnnRealUserDataEndsTest do
            self.uspConfusion:add(nll_acts[self.opt.lstmHist][2][self.rnnRealUserDataEndsTest[i]],
                                self.rnnRealUserDataRewardsTest[self.rnnRealUserDataEndsTest[i]][self.opt.lstmHist])
            _logLoss_score = _logLoss_score + -1 * nll_acts[self.opt.lstmHist][2][self.rnnRealUserDataEndsTest[i]][self.rnnRealUserDataRewardsTest[self.rnnRealUserDataEndsTest[i]][self.opt.lstmHist]]
        end
        self.uspConfusion:updateValids()
        local tvalidScore = self.uspConfusion.totalValid
        self.uspConfusion:zero()

        return {tvalidAct, _logLoss_act/#self.rnnRealUserDataStatesTest, tvalidScore, _logLoss_score/#self.rnnRealUserDataEndsTest}

    elseif string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then
        -- SharedLayer == 1, cnn models
        self.model:evaluate()

        local prepUserState = torch.Tensor(#self.cnnRealUserDataStatesTest, self.opt.lstmHist, self.ciUserSimulator.userStateFeatureCnt)
        for i=1, #self.cnnRealUserDataStatesTest do
            prepUserState[i] = self.ciUserSimulator:preprocessUserStateData(self.cnnRealUserDataStatesTest[i], self.opt.prepro)
        end
        if self.opt.gpu > 0 then
            prepUserState = prepUserState:cuda()
        end
        local nll_acts = self.model:forward(prepUserState)
        nn.utils.recursiveType(nll_acts, 'torch.FloatTensor')
        if nll_acts[1]:ne(nll_acts[1]):sum() > 0 then print('nan appears in output!') os.exit() end
        if nll_acts[2]:ne(nll_acts[2]):sum() > 0 then print('nan appears in output!') os.exit() end

        --- Action prediction evaluation
        self.uapConfusion:zero()
        for i=1, #self.cnnRealUserDataStatesTest do
            self.uapConfusion:add(nll_acts[1][i], self.cnnRealUserDataActsTest[i])
            _logLoss_act = _logLoss_act + -1 * nll_acts[1][i][self.cnnRealUserDataActsTest[i]]
        end
        self.uapConfusion:updateValids()
        local tvalidAct = self.uapConfusion.totalValid
        self.uapConfusion:zero()

        --- Score prediction evaluation
        self.uspConfusion:zero()
        for i=1, #self.cnnRealUserDataEndsTest do
            self.uspConfusion:add(nll_acts[2][self.cnnRealUserDataEndsTest[i]],
                                    self.cnnRealUserDataRewardsTest[self.cnnRealUserDataEndsTest[i]])
            _logLoss_score = _logLoss_score + -1 * nll_acts[2][self.cnnRealUserDataEndsTest[i]][self.cnnRealUserDataRewardsTest[self.cnnRealUserDataEndsTest[i]]]
        end
        self.uspConfusion:updateValids()
        local tvalidScore = self.uspConfusion.totalValid
        self.uspConfusion:zero()

        return {tvalidAct, _logLoss_act/#self.cnnRealUserDataStatesTest, tvalidScore, _logLoss_score/#self.cnnRealUserDataEndsTest}

    else
        -- SharedLayer == 1, non-rnn, non-cnn models
        self.model:evaluate()

        local prepUserState = torch.Tensor(#self.ciUserSimulator.realUserDataStatesTest, self.ciUserSimulator.userStateFeatureCnt)
        for i=1, #self.ciUserSimulator.realUserDataStatesTest do
            prepUserState[i] = self.ciUserSimulator:preprocessUserStateData(self.ciUserSimulator.realUserDataStatesTest[i], self.opt.prepro)
        end
        if self.opt.gpu > 0 then
            prepUserState = prepUserState:cuda()
        end
        local nll_acts = self.model:forward(prepUserState)
        nn.utils.recursiveType(nll_acts, 'torch.FloatTensor')

        if self.opt.uppModel == 'moe' then
            nll_acts = nll_acts:split(self.ciUserSimulator.CIFr.usrActInd_end, 2)  -- We assume 1st dim is batch index. Act pred is the 1st set of output, having dim of 15. Score dim 2.
        end
        if nll_acts[1]:ne(nll_acts[1]):sum() > 0 then print('nan appears in output!') os.exit() end
        if nll_acts[2]:ne(nll_acts[2]):sum() > 0 then print('nan appears in output!') os.exit() end

        --- Action prediction evaluation
        self.uapConfusion:zero()
        for i=1, #self.ciUserSimulator.realUserDataStatesTest do
            self.uapConfusion:add(nll_acts[1][i], self.ciUserSimulator.realUserDataActsTest[i])
            _logLoss_act = _logLoss_act + -1 * nll_acts[1][i][self.ciUserSimulator.realUserDataActsTest[i]]
        end
        self.uapConfusion:updateValids()
        local tvalidAct = self.uapConfusion.totalValid
        self.uapConfusion:zero()

        --- Score prediction evaluation
        self.uspConfusion:zero()
        for i=1, #self.ciUserSimulator.realUserDataEndLinesTest do
            self.uspConfusion:add(nll_acts[2][self.ciUserSimulator.realUserDataEndLinesTest[i]],
                                  self.ciUserSimulator.realUserDataRewardsTest[self.ciUserSimulator.realUserDataEndLinesTest[i]])
            _logLoss_score = _logLoss_score + -1 * nll_acts[2][self.ciUserSimulator.realUserDataEndLinesTest[i]][self.ciUserSimulator.realUserDataRewardsTest[self.ciUserSimulator.realUserDataEndLinesTest[i]]]
        end
        self.uspConfusion:updateValids()
        local tvalidScore = self.uspConfusion.totalValid
        self.uspConfusion:zero()

        -- return a table, with [1] being action pred accuracy, [2] being reward pred accuracy
        return {tvalidAct, _logLoss_act/#self.ciUserSimulator.realUserDataStatesTest, tvalidScore, _logLoss_score/#self.ciUserSimulator.realUserDataEndLinesTest}
    end
end

return CIUserActScorePredictor
