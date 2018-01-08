---
--- Created by pwang8.
--- DateTime: 12/9/17 4:10 PM
--- In this script, we implemented a player outcome predictor with extra soft-labels.
--- This means we use two resources as prediction labels. One is the player outcome
--- binary label, which is the same as the label in UserScorePredictor. The other is
--- the real-valued normalized learning gain difference from the median split value.
--- This script is implemented following UserScorePredictor.lua
--- Linear model is not supported because linear models do not have shared params,
--- so the outcome prediction regression part has no effect on classification training at all.
---

require 'torch'
require 'nn'
require 'optim'
require 'rnn'
local nninit = require 'nninit'
local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'
local OptimMisc = require 'MyMisc.OptimMisc'    -- required to do gradient clipping for rnn modeling training

local CIUserScoreSoftPredictor = classic.class('UserScoreSoftPredictor')

function CIUserScoreSoftPredictor:_init(CIUserSimulator, opt)

    if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
        error('LBFGS should not be used with small mini-batches; 1000 is recommended')
    end

    self.ciUserSimulator = CIUserSimulator
    self.opt = opt
    ----------------------------------------------------------------------
    -- define model to train
    -- on the 2-class (player outcomes with high/low performance) classification problem
    -- with soft-label, which is the exact real-valued outcome score
    self.classes = {}
    for i=1,2 do self.classes[i] = i end
    self.inputFeatureNum = CIUserSimulator.realUserDataStates[1]:size()[1]

    if opt.ciunet == '' then
        -- define model to train
        self.model = nn.Sequential()

        if opt.uppModel == 'moe' then
            ------------------------------------------------------------
            -- mixture of experts
            ------------------------------------------------------------
            local experts = nn.ConcatTable()
            local numOfExp = opt.moeExpCnt
            for i = 1, numOfExp do
                local expert = nn.Sequential()
                expert:add(nn.Reshape(self.inputFeatureNum))
                if opt.dropoutUSim > 0 then expert:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any
                -- The following code creates two output modules. One output module represents predicted
                -- player outcome (binary) classification result, the other represents real-valued outcome
                local mulOutConcatTab = nn.ConcatTable()
                local scoreClsNN = nn.Sequential()  -- score prediction classification module
                scoreClsNN:add(nn.Linear(self.inputFeatureNum, #self.classes))
                scoreClsNN:add(nn.LogSoftMax())
                local scoreRegNN = nn.Sequential()  -- score prediction regression module
                scoreRegNN:add(nn.Linear(self.inputFeatureNum, 1))
                mulOutConcatTab:add(scoreClsNN)
                mulOutConcatTab:add(scoreRegNN) -- {score_classification_output, score_regression_output}
                expert:add(mulOutConcatTab)
                expert:add(nn.JoinTable(-1))    -- this JoinTable will merge the output from a table of 2 elements into a single tensor
                experts:add(expert)
            end

            local gater = nn.Sequential()
            gater:add(nn.Reshape(self.inputFeatureNum))
            if opt.dropoutUSim > 0 then gater:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any
            gater:add(nn.Linear(self.inputFeatureNum, numOfExp))
            gater:add(nn.SoftMax())

            local trunk = nn.ConcatTable()
            trunk:add(gater)
            trunk:add(experts)

            self.model:add(trunk)
            self.model:add(nn.MixtureTable())   -- {gater, experts} is the form of input for MixtureTable. So, gater output should be the 1st in the output table
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
            -- The following code creates two output modules, one is score prediction classification output,
            -- the other one is score prediction regression output.
            local mulOutConcatTab = nn.ConcatTable()
            local scoreClsNN = nn.Sequential()
            scoreClsNN:add(nn.Linear(24, #self.classes))
            scoreClsNN:add(nn.LogSoftMax())
            local scoreRegNN = nn.Sequential()
            scoreRegNN:add(nn.Linear(24, 1))
            mulOutConcatTab:add(scoreClsNN)
            mulOutConcatTab:add(scoreRegNN) -- {score_classification_output, score_regression_output}
            self.model:add(mulOutConcatTab)
            ------------------------------------------------------------

        elseif opt.uppModel == 'rnn_lstm' then
            ------------------------------------------------------------
            -- lstm implementation from Element-Research rnn lib. The lazy dropout (variational RNN models) seems not very correct.
            ------------------------------------------------------------
            self.model:add(nn.Reshape(self.inputFeatureNum))
            --nn.FastLSTM.bn = true
            -- Attention: This lazy dropout seems a little weird. I tried it in player action prediction, but this lazy dropout
            -- hurts the performance of the model. Actually I tried the variational RNN model implemented following RHN and Yarin Gal's
            -- git repo, and that dropout works well.
            local lstm
            if opt.uSimGru == 0 then
                lstm = nn.FastLSTM(self.inputFeatureNum, opt.rnnHdSizeL1, opt.uSimLstmBackLen, nil, nil, nil, opt.dropoutUSim) -- the 3rd param, [rho], the maximum amount of backpropagation steps to take back in time, default value is 9999
                TableSet.fastLSTMForgetGateInit(lstm, opt.dropoutUSim, opt.rnnHdSizeL1, nninit)
            else
                lstm = nn.GRU(self.inputFeatureNum, opt.rnnHdSizeL1, opt.uSimLstmBackLen, opt.dropoutUSim)
            end
            lstm:remember('both')
            self.model:add(lstm)
            self.model:add(nn.NormStabilizer())
            -- if need a 2nd lstm layer
            if opt.rnnHdSizeL2 ~= 0 then
                local lstmL2
                if opt.uSimGru == 0 then
                    lstmL2 = nn.FastLSTM(opt.rnnHdSizeL1, opt.rnnHdSizeL2, opt.uSimLstmBackLen, nil, nil, nil, opt.dropoutUSim) -- the 3rd param, [rho], the maximum amount of backpropagation steps to take back in time, default value is 9999
                    TableSet.fastLSTMForgetGateInit(lstmL2, opt.dropoutUSim, opt.rnnHdSizeL2, nninit)  -- The current implementation is not very flexible
                else
                    lstmL2 = nn.GRU(opt.rnnHdSizeL1, opt.rnnHdSizeL2, opt.uSimLstmBackLen, opt.dropoutUSim)
                end
                lstmL2:remember('both')
                self.model:add(lstmL2)
                self.model:add(nn.NormStabilizer())
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
            local scoreClsNN = nn.Sequential()
            scoreClsNN:add(nn.Linear(lastHidNum, #self.classes))
            scoreClsNN:add(nn.LogSoftMax())
            local scoreRegNN = nn.Sequential()
            scoreRegNN:add(nn.Linear(lastHidNum, 1))
            mulOutConcatTab:add(scoreClsNN)
            mulOutConcatTab:add(scoreRegNN) -- {score_classification_output, score_regression_output}
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
            -- The following code creates two output modules. One module predicts score classification
            -- result, the other module predicts score regression result
            local mulOutConcatTab = nn.ConcatTable()
            local scoreClsNN = nn.Sequential()
            scoreClsNN:add(nn.Linear(opt.rnnHdSizeL1, #self.classes))
            scoreClsNN:add(nn.LogSoftMax())
            local scoreRegNN = nn.Sequential()
            scoreRegNN:add(nn.Linear(opt.rnnHdSizeL1, 1))
            mulOutConcatTab:add(scoreClsNN)
            mulOutConcatTab:add(scoreRegNN) -- {score_classification_output, score_regression_output}
            self.model:add(mulOutConcatTab)
            self.model = nn.Sequencer(self.model)
            ------------------------------------------------------------

        elseif opt.uppModel == 'rnn_rhn_moe' then
            ------------------------------------------------------------
            -- Recurrent Highway Network (dropout mask defined outside rnn model) with MOE head
            ------------------------------------------------------------
            require 'modules.RecurrenHighwayNetworkRNN'
            local rhn
            rhn = nn.RHN(self.inputFeatureNum, opt.rnnHdSizeL1, opt.rhnReccDept, opt.rnnHdLyCnt, opt.uSimLstmBackLen, opt.rnnResidual) --inputSize, outputSize, recurrence_depth, rhn_layers, rho, rnnResidual
            rhn:remember('both')
            self.model:add(rhn)
            self.model:add(nn.NormStabilizer())

            --- moe part
            local experts = nn.ConcatTable()
            local numOfExp = opt.moeExpCnt
            for i = 1, numOfExp do
                local expert = nn.Sequential()
                expert:add(nn.Reshape(opt.rnnHdSizeL1))
                -- The following code creates two output modules. One module predicts score classification
                -- result, the other module predicts score regression result
                local mulOutConcatTab = nn.ConcatTable()
                local scoreClsNN = nn.Sequential()
                scoreClsNN:add(nn.Linear(opt.rnnHdSizeL1, #self.classes))
                scoreClsNN:add(nn.LogSoftMax())
                local scoreRegNN = nn.Sequential()
                scoreRegNN:add(nn.Linear(opt.rnnHdSizeL1, 1))
                mulOutConcatTab:add(scoreClsNN)
                mulOutConcatTab:add(scoreRegNN) -- {score_classification_output, score_regression_output}
                expert:add(mulOutConcatTab)
                expert:add(nn.JoinTable(-1))    -- this JoinTable will merge the output from a table of 2 elements into a single tensor
                experts:add(expert)
            end

            local gater = nn.Sequential()
            gater:add(nn.Reshape(opt.rnnHdSizeL1))
            gater:add(nn.Linear(opt.rnnHdSizeL1, numOfExp))
            gater:add(nn.SoftMax())

            local trunk = nn.ConcatTable()
            trunk:add(gater)
            trunk:add(experts)

            self.model:add(trunk)
            self.model:add(nn.MixtureTable())   -- {gater, experts} is the form of input for MixtureTable. So, gater output should be the 1st in the output table
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
            -- The following code creates two output modules, one module for score classification,
            -- the other for score regression
            local mulOutConcatTab = nn.ConcatTable()
            local scoreClsNN = nn.Sequential()
            scoreClsNN:add(nn.Linear(opt.rnnHdSizeL1, #self.classes))
            scoreClsNN:add(nn.LogSoftMax())
            local scoreRegNN = nn.Sequential()
            scoreRegNN:add(nn.Linear(opt.rnnHdSizeL1, 1))
            mulOutConcatTab:add(scoreClsNN)
            mulOutConcatTab:add(scoreRegNN) -- {score_classification_output, score_regression_output}
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
            local scoreClsNN = nn.Sequential()
            scoreClsNN:add(nn.Linear(self.inputFeatureNum, #self.classes))
            scoreClsNN:add(nn.LogSoftMax())
            local scoreRegNN = nn.Sequential()
            scoreRegNN:add(nn.Linear(self.inputFeatureNum, 1))
            mulOutConcatTab:add(scoreClsNN)
            mulOutConcatTab:add(scoreRegNN) -- {score_classification_output, score_regression_output}
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
            local scoreClsNN = nn.Sequential()
            scoreClsNN:add(nn.Linear(self.inputFeatureNum * opt.lstmHist, #self.classes))
            scoreClsNN:add(nn.LogSoftMax())
            local scoreRegNN = nn.Sequential()
            scoreRegNN:add(nn.Linear(self.inputFeatureNum * opt.lstmHist, 1))
            mulOutConcatTab:add(scoreClsNN)
            mulOutConcatTab:add(scoreRegNN) -- {score_classification_output, score_regression_output}
            self.model:add(mulOutConcatTab)
            ------------------------------------------------------------

        elseif opt.uppModel == 'cnn_uSimCnn_moe' then
            ------------------------------------------------------------
            -- CNN with MOE model
            ------------------------------------------------------------
            require 'modules.TempConvInUserSimCNN'
            local tempCnn = nn.TempConvUserSimCNN()         -- inputSize, outputSize, cnn_layers, kernel_width, dropout_rate, version
            local _tempCnnLayer = tempCnn:CreateCNNModule(self.inputFeatureNum, self.inputFeatureNum, opt.rnnHdLyCnt, opt.cnnKernelWidth, opt.dropoutUSim, opt.lstmHist, opt.cnnConnType)
            self.model:add(_tempCnnLayer)
            self.model:add(nn.View(-1):setNumInputDims(2))  -- The input/output data should have dimensions of batch_index/frame_index/feature_index, so it's 3d, and 2d without batch index

            --- moe part
            local experts = nn.ConcatTable()
            local numOfExp = opt.moeExpCnt
            for i = 1, numOfExp do
                local expert = nn.Sequential()
                expert:add(nn.Reshape(self.inputFeatureNum * opt.lstmHist))
                if opt.dropoutUSim > 0 then expert:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any
                -- The following code creates two output modules. One output module represents predicted
                -- player outcome (binary) classification result, the other represents real-valued outcome
                local mulOutConcatTab = nn.ConcatTable()
                local scoreClsNN = nn.Sequential()  -- score prediction classification module
                scoreClsNN:add(nn.Linear(self.inputFeatureNum * opt.lstmHist, #self.classes))
                scoreClsNN:add(nn.LogSoftMax())
                local scoreRegNN = nn.Sequential()  -- score prediction regression module
                scoreRegNN:add(nn.Linear(self.inputFeatureNum * opt.lstmHist, 1))
                mulOutConcatTab:add(scoreClsNN)
                mulOutConcatTab:add(scoreRegNN) -- {score_classification_output, score_regression_output}
                expert:add(mulOutConcatTab)
                expert:add(nn.JoinTable(-1))    -- this JoinTable will merge the output from a table of 2 elements into a single tensor
                experts:add(expert)
            end

            local gater = nn.Sequential()
            gater:add(nn.Reshape(self.inputFeatureNum * opt.lstmHist))
            if opt.dropoutUSim > 0 then gater:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any
            gater:add(nn.Linear(self.inputFeatureNum * opt.lstmHist, numOfExp))
            gater:add(nn.SoftMax())

            local trunk = nn.ConcatTable()
            trunk:add(gater)
            trunk:add(experts)

            self.model:add(trunk)
            self.model:add(nn.MixtureTable())   -- {gater, experts} is the form of input for MixtureTable. So, gater output should be the 1st in the output table
            ------------------------------------------------------------

        else
            print('Unknown uppModel type'..opt.uppModel..' in UserScoreSoftPredictor training')
            torch.CmdLine():text()
            error()
        end

        -- params init
        local uspLinearLayers = self.model:findModules('nn.Linear')
        for l = 1, #uspLinearLayers do
            uspLinearLayers[l]:init('weight', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)}):init('bias', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)})
        end
    elseif opt.ciunet == 'rlLoad' then  -- If need reload a trained usp model in the RL training/evaluation, not for training usp anymore
        if string.sub(opt.uppModelUsp, 1, 7) == 'rnn_rhn' then
            require 'modules.RecurrenHighwayNetworkRNN'
        elseif opt.uppModelUsp == 'rnn_blstm' then
            require 'modules.LSTMBayesianRNN'
        elseif opt.uppModelUsp == 'rnn_bGridlstm' then
            require 'modules.GridLSTMBayesianRNN'
        elseif string.sub(opt.uppModelUsp, 1, 4) == 'cnn_' then
            require 'modules.TempConvInUserSimCNN'
        end
        self.model = torch.load(paths.concat(opt.ubgDir , opt.uspFile))
    else
        print('<trainer> reloading previously trained ciunet')
        self.model = torch.load(opt.ciunet)
    end

    --    -- verbose
    --    print(self.model)

    ----------------------------------------------------------------------
    -- loss function: negative log-likelihood for score classification
    -- and mean squared error for score regression
    self.uspPrlCriterion = nn.ParallelCriterion()
    self.uspClsCriterion = nn.ClassNLLCriterion()
    self.uspRegCriterion = nn.MSECriterion()
    self.uspPrlCriterion:add(self.uspClsCriterion)   -- score (outcome) prediction classification loss. The weight for this classification error source is 1
    self.uspPrlCriterion:add(self.uspRegCriterion, self.opt.uSimScSoft)   -- score (outcome) prediction regression loss, opt.uSimScSoft is the weight for this error source
    if string.sub(opt.uppModel, 1, 4) == 'rnn_' then
        self.uspPrlCriterion = nn.SequencerCriterion(self.uspPrlCriterion)
    end

    self.trainEpoch = 1
    -- this matrix records the current confusion across classes
    self.uspConfusion = optim.ConfusionMatrix(self.classes)

    -- log results to files
    self.uspTrainLogger = optim.Logger(paths.concat('userModelTrained', opt.save, 'scrSoft_train.log'))
    self.uspTestLogger = optim.Logger(paths.concat('userModelTrained', opt.save, 'scrSoft_test.log'))
    self.uspTestLogger:setNames{'Epoch', 'Score Test acc.', 'Score Test LogLoss'}
    self.uspClsRegLogger = optim.Logger(paths.concat('userModelTrained', opt.save, 'scrSoft_testClsReg.log'))
    self.uspClsRegLogger:setNames{'Epoch', 'Rl_nlg', 'Correct', 'Ls_1', 'Ls_2', 'Est_nlg'}

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
            self.uspPrlCriterion = self.uspPrlCriterion:cuda()
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
    self.rnnRealUserDataStandardNLGTest = self.ciUserSimulator.rnnRealUserDataStandardNLGTest
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
    self.cnnRealUserDataStandardNLGTest = self.ciUserSimulator.cnnRealUserDataStandardNLGTest
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
    self.uspParam, self.uspDParam = self.model:getParameters()
end

-- training function
function CIUserScoreSoftPredictor:trainOneEpoch()
    -- local vars
    local time = sys.clock()

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. self.trainEpoch .. ' [batchSize = ' .. self.opt.batchSize .. ']')
    local inputs
    local targetsClsRegScore = {}   -- the output of NN contains 2 elements in a table
    local targets   -- targets are player score labels (classification) in this script
    local plh_acts  -- place holder for player action label.
    local score_reg -- the regression target for score/outcome prediction output
    local closeToEnd
    local t = 1
    local lstmIter = 1  -- lstm iterate for each squence starts from this value
    local epochDone = false
    while not epochDone do
        if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
            -- rnn models
            inputs = {}
            targets = {}
            plh_acts = {}
            score_reg = {}
            closeToEnd = torch.Tensor(self.opt.batchSize):fill(0)
            local k
            for j = 1, self.opt.lstmHist do
                inputs[j] = torch.Tensor(self.opt.batchSize, self.inputFeatureNum)
                targets[j] = torch.Tensor(self.opt.batchSize)
                plh_acts[j] = torch.Tensor(self.opt.batchSize)
                score_reg[j] = torch.Tensor(self.opt.batchSize)
                k = 1
                for i = lstmIter, math.min(lstmIter+self.opt.batchSize-1, #self.rnnRealUserDataStates) do
                    inputs[j][k] = self.rnnRealUserDataStates[i][j]
                    targets[j][k] = self.rnnRealUserDataRewards[i][j]
                    plh_acts[j][k] = self.rnnRealUserDataActs[i][j]
                    score_reg[j][k] = self.rnnRealUserDataStandardNLG[i][j] -- the score/outcome regression prediction ground-truth
                    if j == self.opt.lstmHist then
                        for dis=0, self.opt.scorePredStateScope-1 do
                            if (i+dis) <= #self.rnnRealUserDataStates and TableSet.tableContainsValue(self.rnnRealUserDataEnds, i+dis) then
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
                        targets[j][k] = self.rnnRealUserDataRewards[randInd][j]
                        plh_acts[j][k] = self.rnnRealUserDataActs[randInd][j]
                        score_reg[j][k] = self.rnnRealUserDataStandardNLG[randInd][j]
                        if j == self.opt.lstmHist then
                            for dis=0, self.opt.scorePredStateScope-1 do
                                if (randInd+dis) <= #self.rnnRealUserDataStates and TableSet.tableContainsValue(self.rnnRealUserDataEnds, randInd+dis) then
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
                self.ciUserSimulator:UserSimActDataAugment(inputs, plh_acts, targets, score_reg, self.opt.uppModel)
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
                nn.utils.recursiveType(targets, 'torch.CudaTensor')
                nn.utils.recursiveType(score_reg, 'torch.CudaTensor')
                closeToEnd = closeToEnd:cuda()
                -- do not need to transform plh_acts into cudaTensor because it is not actually used in training. It is used in data augmentation.
            end

            -- Put score/outcome classification and regression targets into a table, to make it match the format of the multi-task NN output
            for j = 1, self.opt.lstmHist do
                targetsClsRegScore[j] = {targets[j], score_reg[j]}
            end

        elseif string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then
            -- cnn models
            inputs = torch.Tensor(self.opt.batchSize, self.opt.lstmHist, self.inputFeatureNum)
            targets = torch.Tensor(self.opt.batchSize)  -- labels for score classification
            plh_acts = torch.Tensor(self.opt.batchSize) -- labels for player action
            score_reg = torch.Tensor(self.opt.batchSize)  -- standard nlg for score regression
            closeToEnd = torch.Tensor(self.opt.batchSize):fill(0)

            local k = 1
            for i = t, math.min(t+self.opt.batchSize-1, #self.cnnRealUserDataStates) do
                inputs[k] = self.cnnRealUserDataStates[i]
                targets[k] = self.cnnRealUserDataRewards[i]
                plh_acts[k] = self.cnnRealUserDataActs[i]
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
                    -- I'll put input pre-process after data augmentation
                    inputs[k] = self.cnnRealUserDataStates[randInd]
                    targets[k] = self.cnnRealUserDataRewards[randInd]
                    plh_acts[k] = self.cnnRealUserDataActs[randInd]
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
                self.ciUserSimulator:UserSimActDataAugment(inputs, plh_acts, targets, score_reg, self.opt.uppModel)
            end
            -- Should do input feature pre-processing after data augmentation
            inputs = self.ciUserSimulator:preprocessUserStateData(inputs, self.opt.prepro)
            -- Try to add random normal noise to input features and see how it performs
            -- This should be invoked after input preprocess bcz we want to set an unique std
            --self.ciUserSimulator:UserSimDataAddRandNoise(inputs, false, 0.01)

            if self.opt.gpu > 0 then
                inputs = inputs:cuda()
                targets = targets:cuda()
                score_reg = score_reg:cuda()
                closeToEnd = closeToEnd:cuda()
            end

            targetsClsRegScore = {targets, score_reg}

        else
            -- non-rnn, non-cnn models, create mini batch
            inputs = torch.Tensor(self.opt.batchSize, self.inputFeatureNum)
            targets = torch.Tensor(self.opt.batchSize)      -- score/outcome classification labels
            plh_acts = torch.Tensor(self.opt.batchSize)
            score_reg = torch.Tensor(self.opt.batchSize)    -- score/outcome regression ground-truth
            closeToEnd = torch.Tensor(self.opt.batchSize):fill(0)
            local k = 1
            for i = t, math.min(t+self.opt.batchSize-1, #self.ciUserSimulator.realUserDataStates) do
                inputs[k] = self.ciUserSimulator.realUserDataStates[i]
                targets[k] = self.ciUserSimulator.realUserDataRewards[i]
                plh_acts[k] = self.ciUserSimulator.realUserDataActs[i]
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
                    targets[k] = self.ciUserSimulator.realUserDataRewards[randInd]
                    plh_acts[k] = self.ciUserSimulator.realUserDataActs[randInd]
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
                self.ciUserSimulator:UserSimActDataAugment(inputs, plh_acts, targets, score_reg, self.opt.uppModel)
            end
            -- Should do input feature pre-processing after data augmentation
            inputs = self.ciUserSimulator:preprocessUserStateData(inputs, self.opt.prepro)
            -- Try to add random normal noise to input features and see how it performs
            -- This should be invoked after input preprocess bcz we want to set an unique std
            --self.ciUserSimulator:UserSimDataAddRandNoise(inputs, false, 0.01)

            if self.opt.gpu > 0 then
                inputs = inputs:cuda()
                targets = targets:cuda()
                score_reg = score_reg:cuda()
                closeToEnd = closeToEnd:cuda()
            end

            targetsClsRegScore = {targets, score_reg}

        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- just in case:
            collectgarbage()

            -- get new parameters
            if x ~= self.uspParam then
                self.uspParam:copy(x)
            end

            -- reset gradients
            self.uspDParam:zero()

            -- evaluate function for complete mini batch
            local outputs = self.model:forward(inputs)

            -- Here, if moe is used with shared lower layers, it is the problem that,
            -- due to limitation of MixtureTable module, we have to join tables together as
            -- a single tensor as output of the user score classification and regression prediction models.
            -- So, to guarantee the compatability, we need split the tensor into two tables here,
            -- for score classification and regression prediction respectively.
            if string.sub(self.opt.uppModel, -3, -1) == 'moe' then
                if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
                    for i=1, #outputs do
                        outputs[i] = outputs[i]:split(#self.classes, 2)  -- We assume 1st dim is batch index. Score classification is the 1st set of output, having 2 output values. Score regression has 1 output.
                    end
                else
                    outputs = outputs:split(#self.classes, 2)  -- We assume 1st dim is batch index. Score classification is the 1st set of output, having 2 output values. Score regression has 1 output.
                end
            end

            -- Zero error values (change output to target) for score prediction cases far away from ending state (I don't hope these cases influence training)
            for cl=1, closeToEnd:size(1) do
                if closeToEnd[cl] < 1 then
                    if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
                        outputs[self.opt.lstmHist][1][cl][targetsClsRegScore[self.opt.lstmHist][1][cl]] = 0 -- Attention: because LogSoftMax is the output of a classification module, and NLL is criterion, so if we set ground-truth labeled output into 0, this will generate no error
                        outputs[self.opt.lstmHist][2][cl] = targetsClsRegScore[self.opt.lstmHist][2][cl]    -- Because this is the socre regression part, we only need to set the output to groud-truth to make no error
                    else
                        outputs[1][cl][targetsClsRegScore[1][cl]] = 0
                        outputs[2][cl] = targetsClsRegScore[2][cl]
                    end
                end
            end
            local f = self.uspPrlCriterion:forward(outputs, targetsClsRegScore)
            local df_do = self.uspPrlCriterion:backward(outputs, targetsClsRegScore)

            if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
                f = self.uspPrlCriterion:forward({outputs[self.opt.lstmHist]}, {targetsClsRegScore[self.opt.lstmHist]})
                for step=1, self.opt.lstmHist-1 do
                    assert(type(df_do)=='table' and #df_do==self.opt.lstmHist, 'df_do format is not correct!')
                    df_do[step][1]:zero()   -- zero df_do for both score classification and score regression
                    df_do[step][2]:zero()   -- for time steps before the last one
                end
            end

            -- If moe with shared lower layers, we merge the df_do before backprop. This is necessary because the network has merged classification and regression output in one tensor
            if string.sub(self.opt.uppModel, -3, -1) == 'moe' then
                if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
                    -- rnn with moe models
                    for i=1, #df_do do
                        df_do[i] = nn.JoinTable(-1):forward(df_do[i])  -- We assume 1st dim is batch index.
                    end
                    if self.opt.gpu > 0 then
                        nn.utils.recursiveType(df_do, 'torch.CudaTensor')
                    end
                else
                    -- cnn or non-rnn, non-cnn models but with moe output
                    df_do = nn.JoinTable(-1):forward(df_do)  -- We assume 1st dim is batch index.
                    if self.opt.gpu > 0 then
                        df_do = df_do:cuda()  -- seems like after calling the forward function of nn.JoinTable, that output(df_do) becomes main memory object again
                    end
                end
            end

            self.model:backward(inputs, df_do)

            -- penalties (L1 and L2):
            if self.opt.coefL1 ~= 0 or self.opt.coefL2 ~= 0 then
                -- locals:
                local norm,sign= torch.norm,torch.sign

                -- Loss:
                f = f + self.opt.coefL1 * norm(self.uspParam,1)
                f = f + self.opt.coefL2 * norm(self.uspParam,2)^2/2

                -- Gradients:
                self.uspDParam:add( sign(self.uspParam):mul(self.opt.coefL1) + self.uspParam:clone():mul(self.opt.coefL2) )
            end

            -- update self.uspConfusion
            if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
                for i = 1,self.opt.batchSize do
                    self.uspConfusion:add(outputs[self.opt.lstmHist][1][i], targetsClsRegScore[self.opt.lstmHist][1][i])
                end
            else
                -- for cnn and non-rnn, non-cnn models
                for i = 1,self.opt.batchSize do
                    self.uspConfusion:add(outputs[1][i], targetsClsRegScore[1][i])
                end
            end

            -- gradient clipping. It is recommended for rnn models, not sure if it is helpful to other models.
            if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
                OptimMisc.clipGradByNorm(self.uspDParam, 10)    -- right now 10 is used constantly as clipping norm
            end
            -- return f and df/dX
            return f, self.uspDParam
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
            optim.lbfgs(feval, self.uspParam, lbfgsState)

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
            optim.sgd(feval, self.uspParam, sgdState)

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
            optim.adam(feval, self.uspParam, adamState)

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
            optim.rmsprop(feval, self.uspParam, rmspropState)

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
    --    time = time / #self.ciUserSimulator.realUserDataStates  -- This is not accurate, if lstm is used
    print("<trainer> time to learn 1 epoch = " .. (time*1000) .. 'ms')

    -- print self.uspConfusion matrix
    --    print(self.uspConfusion)
    self.uspConfusion:updateValids()
    local confMtxStr = 'Score classification average row correct: ' .. (self.uspConfusion.averageValid*100) .. '% \n' ..
    'average rowUcol correct (VOC measure): ' .. (self.uspConfusion.averageUnionValid*100) .. '% \n' ..
    ' + global correct: ' .. (self.uspConfusion.totalValid*100) .. '%'
    print(confMtxStr)
    self.uspTrainLogger:add{['% mean class accuracy (train set)'] = self.uspConfusion.totalValid * 100}


    -- save/log current net
    local filename = paths.concat('userModelTrained', self.opt.save, 'uspSoft.t7')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    --    if paths.filep(filename) then
    --        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    --    end

    if self.trainEpoch % 10 == 0 and self.opt.ciuTType == 'train' then
        filename = paths.concat('userModelTrained', self.opt.save, string.format('%d', self.trainEpoch)..'_'..string.format('%.2f', self.uspConfusion.totalValid*100)..'_uspSoft.t7')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('<trainer> saving periodly trained ciunet to '..filename)
        torch.save(filename, self.model)
    end

    if self.trainEpoch == self.opt.usimTrIte and self.opt.ciuTType == 'train' then
        -- Save the trained model after the final epoch
        filename = paths.concat('userModelTrained', self.opt.save, string.format('final_%d', self.trainEpoch)..'_'..string.format('%.2f', self.uspConfusion.totalValid*100)..'_uspSoft.t7')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('<trainer> saving final trained action prediction ciunet to '..filename)
        torch.save(filename, self.model)
    end

    if (self.opt.ciuTType == 'train' or self.opt.ciuTType == 'train_tr') and self.trainEpoch % self.opt.testOnTestFreq == 0 then
        local testEval = self:testScorePredOnTestDetOneEpoch()
        print('<Score prediction accuracy at epoch '..string.format('%d', self.trainEpoch)..' on test set > '..string.format('%.2f%%', testEval[1]*100)..
        ', and LogLoss '..string.format('%.2f', testEval[2]))
        self.uspTestLogger:add{string.format('%d', self.trainEpoch), string.format('%.5f%%', testEval[1]*100), string.format('%.5f', testEval[2])}
    end

    if (self.opt.ciuTType == 'train' or self.opt.ciuTType == 'train_tr') and self.trainEpoch % self.opt.testOnTestSoftScoreFreq == 0 then
        self:testScoreClsRegOnTestDetOneEpoch()
    end

    self.uspConfusion:zero()
    -- next epoch
    self.trainEpoch = self.trainEpoch + 1
end

-- evaluation function on test/train_validation set
function CIUserScoreSoftPredictor:testScorePredOnTestDetOneEpoch()
    -- just in case:
    collectgarbage()
    -- Confusion matrix for score prediction (2-class)
    --    local scrPredTP = torch.Tensor(2):fill(1e-3)
    --    local scrPredFP = torch.Tensor(2):fill(1e-3)
    --    local scrPredFN = torch.Tensor(2):fill(1e-3)
    local _logLoss = 0
    self.uspConfusion:zero()

    if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
        -- uSimShLayer == 0 and rnn model
        if self.opt.uSimBayesEvl > 0 then
            self.model:training()
        else
            self.model:evaluate()
        end

        local tabStateOrigin = {}
        for j=1, self.opt.lstmHist do
            local prepUserState = torch.Tensor(#self.rnnRealUserDataEndsTest, self.ciUserSimulator.userStateFeatureCnt)
            for k=1, #self.rnnRealUserDataEndsTest do
                prepUserState[k] = self.ciUserSimulator:preprocessUserStateData(self.rnnRealUserDataStatesTest[self.rnnRealUserDataEndsTest[k]][j], self.opt.prepro)
            end
            tabStateOrigin[j] = prepUserState
        end

        -- Evaluation epochs is either self.opt.uSimBayesEvl if it is not 0, or 1
        local evlCnt = (self.opt.uSimBayesEvl > 0 and self.opt.uSimBayesEvl or 1)

        for evl_ite = 1, evlCnt do
            self.model:forget()
            local tabState = {}
            local test_rnn_noise_i = {}
            local test_rnn_noise_h = {}
            local test_rnn_noise_o = {}
            if self.opt.uppModelRNNDom > 0 then
                TableSet.buildRNNDropoutMask(test_rnn_noise_i, test_rnn_noise_h, test_rnn_noise_o, self.inputFeatureNum, self.opt.rnnHdSizeL1, self.opt.rnnHdLyCnt, #self.rnnRealUserDataEndsTest, self.opt.lstmHist, self.opt.uppModelRNNDom)
                if self.opt.uSimBayesEvl > 0 then
                    TableSet.sampleRNNDropoutMask(self.opt.dropoutUSim, test_rnn_noise_i, test_rnn_noise_h, test_rnn_noise_o, self.opt.rnnHdLyCnt, self.opt.lstmHist)
                else
                    TableSet.sampleRNNDropoutMask(0, test_rnn_noise_i, test_rnn_noise_h, test_rnn_noise_o, self.opt.rnnHdLyCnt, self.opt.lstmHist)
                end
                for j = 1, self.opt.lstmHist do
                    tabState[j] = {tabStateOrigin[j], test_rnn_noise_i[j], test_rnn_noise_h[j], test_rnn_noise_o[j]}
                end
            end

            if self.opt.gpu > 0 then
                nn.utils.recursiveType(tabState, 'torch.CudaTensor')
            end
            local nll_rewards = self.model:forward(tabState)
            if string.sub(self.opt.uppModel, -3, -1) == 'moe' then
                nll_rewards[self.opt.lstmHist] = nll_rewards[self.opt.lstmHist]:split(#self.classes, 2)  -- We assume 1st dim is batch index. Score classification is the 1st set of output, having 2 outputs. Score regression is the 2nd set of output.
            end
            nn.utils.recursiveType(nll_rewards, 'torch.FloatTensor')
            if nll_rewards[self.opt.lstmHist][1]:ne(nll_rewards[self.opt.lstmHist][1]):sum() > 0 then print('nan appears in output!') os.exit() end
            if nll_rewards[self.opt.lstmHist][2]:ne(nll_rewards[self.opt.lstmHist][2]):sum() > 0 then print('nan appears in output!') os.exit() end

            for i=1, #self.rnnRealUserDataEndsTest do
                self.uspConfusion:add(nll_rewards[self.opt.lstmHist][1][i], self.rnnRealUserDataRewardsTest[self.rnnRealUserDataEndsTest[i]][self.opt.lstmHist])
                _logLoss = _logLoss + -1 * nll_rewards[self.opt.lstmHist][1][i][self.rnnRealUserDataRewardsTest[self.rnnRealUserDataEndsTest[i]][self.opt.lstmHist]]
            end
        end

        self.uspConfusion:updateValids()
        local tvalid = self.uspConfusion.totalValid
        self.uspConfusion:zero()
        return {tvalid, _logLoss / (#self.rnnRealUserDataEndsTest * evlCnt)}

    elseif string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then
        -- uSimShLayer == 0 and cnn models
        if self.opt.uSimBayesEvl > 0 then
            self.model:training()
        else
            self.model:evaluate()
        end

        local prepUserState = torch.Tensor(#self.cnnRealUserDataEndsTest, self.opt.lstmHist, self.ciUserSimulator.userStateFeatureCnt)
        for i=1, #self.cnnRealUserDataEndsTest do
            prepUserState[i] = self.ciUserSimulator:preprocessUserStateData(self.cnnRealUserDataStatesTest[self.cnnRealUserDataEndsTest[i]], self.opt.prepro)
        end
        if self.opt.gpu > 0 then
            prepUserState = prepUserState:cuda()
        end

        -- Evaluation epochs is either self.opt.uSimBayesEvl if it is not 0, or 1
        local evlCnt = (self.opt.uSimBayesEvl > 0 and self.opt.uSimBayesEvl or 1)

        for evl_ite = 1, evlCnt do
            local nll_rewards = self.model:forward(prepUserState)
            if string.sub(self.opt.uppModel, -3, -1) == 'moe' then
                nll_rewards = nll_rewards:split(#self.classes, 2)  -- We assume 1st dim is batch index. Score classification is the 1st set of output, having 2 outputs. Score regression is the 2nd set of output.
            end
            nn.utils.recursiveType(nll_rewards, 'torch.FloatTensor')
            if nll_rewards[1]:ne(nll_rewards[1]):sum() > 0 then print('nan appears in output!') os.exit() end
            if nll_rewards[2]:ne(nll_rewards[2]):sum() > 0 then print('nan appears in output!') os.exit() end

            for i=1, #self.cnnRealUserDataEndsTest do
                self.uspConfusion:add(nll_rewards[1][i], self.cnnRealUserDataRewardsTest[self.cnnRealUserDataEndsTest[i]])
                _logLoss = _logLoss + -1 * nll_rewards[1][i][self.cnnRealUserDataRewardsTest[self.cnnRealUserDataEndsTest[i]]]
            end
        end

        self.uspConfusion:updateValids()
        local tvalid = self.uspConfusion.totalValid
        self.uspConfusion:zero()
        return {tvalid, _logLoss / (#self.cnnRealUserDataEndsTest * evlCnt)}

    else
        -- uSimShLayer == 0 and non-rnn, non-cnn models
        if self.opt.uSimBayesEvl > 0 then
            self.model:training()
        else
            self.model:evaluate()
        end

        local prepUserState = torch.Tensor(#self.ciUserSimulator.realUserDataEndLinesTest, self.ciUserSimulator.userStateFeatureCnt)
        for i=1, #self.ciUserSimulator.realUserDataEndLinesTest do
            prepUserState[i] = self.ciUserSimulator:preprocessUserStateData(self.ciUserSimulator.realUserDataStatesTest[self.ciUserSimulator.realUserDataEndLinesTest[i]], self.opt.prepro)
        end
        if self.opt.gpu > 0 then
            prepUserState = prepUserState:cuda()
        end

        -- Evaluation epochs is either self.opt.uSimBayesEvl if it is not 0, or 1
        local evlCnt = (self.opt.uSimBayesEvl > 0 and self.opt.uSimBayesEvl or 1)

        for evl_ite = 1, evlCnt do
            local nll_rewards = self.model:forward(prepUserState)
            nn.utils.recursiveType(nll_rewards, 'torch.FloatTensor')

            if string.sub(self.opt.uppModel, -3, -1) == 'moe' then
                nll_rewards = nll_rewards:split(#self.classes, 2)  -- We assume 1st dim is batch index. Score classification is the 1st set of output, having 2 outputs. Score regression is the 2nd set of output.
            end
            if nll_rewards[1]:ne(nll_rewards[1]):sum() > 0 then print('nan appears in output!') os.exit() end
            if nll_rewards[2]:ne(nll_rewards[2]):sum() > 0 then print('nan appears in output!') os.exit() end

            for i=1, #self.ciUserSimulator.realUserDataEndLinesTest do
                self.uspConfusion:add(nll_rewards[1][i], self.ciUserSimulator.realUserDataRewardsTest[self.ciUserSimulator.realUserDataEndLinesTest[i]])
                _logLoss = _logLoss + -1 * nll_rewards[1][i][self.ciUserSimulator.realUserDataRewardsTest[self.ciUserSimulator.realUserDataEndLinesTest[i]]]
            end
        end

        self.uspConfusion:updateValids()
        local tvalid = self.uspConfusion.totalValid
        self.uspConfusion:zero()
        return {tvalid, _logLoss / (#self.ciUserSimulator.realUserDataEndLinesTest * evlCnt)}
    end
end

-- This is the specific evaluation function applied on UserScoreSoftPredictor module
-- We demonstrate how the classification and regression model performs on each data point
-- to see if we can utilize the regression module more in the classification task
function CIUserScoreSoftPredictor:testScoreClsRegOnTestDetOneEpoch()
    -- just in case:
    collectgarbage()

    --- right now only cnn_ models are supported
    if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
        ---- uSimShLayer == 0 and rnn model
    elseif string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then
        -- uSimShLayer == 0 and cnn models
        if self.opt.uSimBayesEvl > 0 then
            self.model:training()
        else
            self.model:evaluate()
        end

        local prepUserState = torch.Tensor(#self.cnnRealUserDataEndsTest, self.opt.lstmHist, self.ciUserSimulator.userStateFeatureCnt)
        for i=1, #self.cnnRealUserDataEndsTest do
            prepUserState[i] = self.ciUserSimulator:preprocessUserStateData(self.cnnRealUserDataStatesTest[self.cnnRealUserDataEndsTest[i]], self.opt.prepro)
        end
        if self.opt.gpu > 0 then
            prepUserState = prepUserState:cuda()
        end

        -- Evaluation epochs is either self.opt.uSimBayesEvl if it is not 0, or 1
        local evlCnt = (self.opt.uSimBayesEvl > 0 and self.opt.uSimBayesEvl or 1)

        for evl_ite = 1, evlCnt do
            local nll_rewards = self.model:forward(prepUserState)
            if string.sub(self.opt.uppModel, -3, -1) == 'moe' then
                nll_rewards = nll_rewards:split(#self.classes, 2)  -- We assume 1st dim is batch index. Score classification is the 1st set of output, having 2 outputs. Score regression is the 2nd set of output.
            end
            nn.utils.recursiveType(nll_rewards, 'torch.FloatTensor')
            if nll_rewards[1]:ne(nll_rewards[1]):sum() > 0 then print('nan appears in output!') os.exit() end
            if nll_rewards[2]:ne(nll_rewards[2]):sum() > 0 then print('nan appears in output!') os.exit() end

            for i=1, #self.cnnRealUserDataEndsTest do
                local _realRewardClsLabel = self.cnnRealUserDataRewardsTest[self.cnnRealUserDataEndsTest[i]]
                local _theOtherClsIdx = (_realRewardClsLabel == 1 and 2 or 1)
                self.uspClsRegLogger:add{
                    string.format('%d', self.trainEpoch),
                    string.format('%.3f', self.cnnRealUserDataStandardNLGTest[self.cnnRealUserDataEndsTest[i]]),
                    string.format('%.3f', (nll_rewards[1][i][_realRewardClsLabel] > nll_rewards[1][i][_theOtherClsIdx] and 1 or 0)),
                    string.format('%.3f', nll_rewards[1][i][1]),
                    string.format('%.3f', nll_rewards[1][i][2]),
                    string.format('%.3f', nll_rewards[2][i][1])}
            end
        end

    else
        ---- uSimShLayer == 0 and non-rnn, non-cnn models
    end
end

return CIUserScoreSoftPredictor