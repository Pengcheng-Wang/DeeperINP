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

local CIUserActScorePredictor = classic.class('UserActScorePredictor')

function CIUserActScorePredictor:_init(CIUserSimulator, opt)

    -- batch size?
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
            experts = nn.ConcatTable()
            local numOfExp = 4
            for i = 1, numOfExp do
                local expert = nn.Sequential()
                expert:add(nn.Linear(self.inputFeatureNum, 32))
                expert:add(nn.ReLU())
                if opt.dropoutUSim > 0 then expert:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any
                expert:add(nn.Linear(32, 24))
                expert:add(nn.ReLU())
                if opt.dropoutUSim > 0 then expert:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any

                -- The following code creates two output modules, with one module matches
                -- to user action prediction, and the other matches to user outcome(score) prediction
                mulOutConcatTab = nn.ConcatTable()
                actSeqNN = nn.Sequential()
                actSeqNN:add(nn.Linear(24, #classesActs))
                actSeqNN:add(nn.LogSoftMax())
                scoreSeqNN = nn.Sequential()
                scoreSeqNN:add(nn.Linear(24, #classesScores))
                scoreSeqNN:add(nn.LogSoftMax())
                mulOutConcatTab:add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
                mulOutConcatTab:add(scoreSeqNN) -- {act, outcome(score)}

                expert:add(mulOutConcatTab)
                expert:add(nn.JoinTable(-1))
                experts:add(expert)
            end

            gater = nn.Sequential()
            gater:add(nn.Linear(self.inputFeatureNum, 24))
            gater:add(nn.Tanh())
            if opt.dropoutUSim > 0 then gater:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any
            gater:add(nn.Linear(24, numOfExp))
            gater:add(nn.SoftMax())

            trunk = nn.ConcatTable()
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
            mulOutConcatTab = nn.ConcatTable()
            actSeqNN = nn.Sequential()
            actSeqNN:add(nn.Linear(24, #classesActs))
            actSeqNN:add(nn.LogSoftMax())
            scoreSeqNN = nn.Sequential()
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
            mulOutConcatTab = nn.ConcatTable()
            actSeqNN = nn.Sequential()
            actSeqNN:add(nn.Linear(self.inputFeatureNum, #classesActs))
            actSeqNN:add(nn.LogSoftMax())
            scoreSeqNN = nn.Sequential()
            scoreSeqNN:add(nn.Linear(self.inputFeatureNum, #classesScores))
            scoreSeqNN:add(nn.LogSoftMax())
            mulOutConcatTab:add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
            mulOutConcatTab:add(scoreSeqNN) -- {act, outcome(score)}

            self.model:add(mulOutConcatTab)
            ------------------------------------------------------------

        elseif opt.uppModel == 'lstm' then
            ------------------------------------------------------------
            -- lstm
            ------------------------------------------------------------
            self.model:add(nn.Reshape(self.inputFeatureNum))
            nn.FastLSTM.bn = true   -- turn on batch normalization
            local lstm
            if opt.uSimGru == 0 then
                lstm = nn.FastLSTM(self.inputFeatureNum, opt.lstmHd, opt.uSimLstmBackLen, nil, nil, nil, opt.dropoutUSim) -- the 3rd param, [rho], the maximum amount of backpropagation steps to take back in time, default value is 9999
                TableSet.fastLSTMForgetGateInit(lstm, opt.dropoutUSim, opt.lstmHd, nninit)
                -- has not applied batch normalization for fastLSTM before, should try it.
            else
                lstm = nn.GRU(self.inputFeatureNum, opt.lstmHd, opt.uSimLstmBackLen, opt.dropoutUSim)   -- did not apply dropout or batchNormalization for GRU before
            end
            lstm:remember('both')
            self.model:add(lstm)
            self.model:add(nn.NormStabilizer())
            -- if need a 2nd lstm layer
            if opt.lstmHdL2 ~= 0 then
                local lstmL2
                if opt.uSimGru == 0 then
                    lstmL2 = nn.FastLSTM(opt.lstmHd, opt.lstmHdL2, opt.uSimLstmBackLen, nil, nil, nil, opt.dropoutUSim) -- the 3rd param, [rho], the maximum amount of backpropagation steps to take back in time, default value is 9999
                    TableSet.fastLSTMForgetGateInit(lstmL2, opt.dropoutUSim, opt.lstmHdL2, nninit)
                    -- has not applied batch normalization for fastLSTM before, should try it.
                else
                    lstmL2 = nn.GRU(opt.lstmHd, opt.lstmHdL2, opt.uSimLstmBackLen, opt.dropoutUSim)     -- did not apply dropout or batchNormalization for GRU before
                end
                lstmL2:remember('both')
                self.model:add(lstmL2)
                self.model:add(nn.NormStabilizer()) -- I am not very clear if NormStabilizer should be used togeher with Dropout, especially since dropout is used on memory value, equals to change memory value distribution a little
            end
            local lastHidNum
            if opt.lstmHdL2 == 0 then
                lastHidNum = opt.lstmHd
            else
                lastHidNum = opt.lstmHdL2
            end

            -- The following code creates two output modules, with one module matches
            -- to user action prediction, and the other matches to user outcome(score) prediction
            mulOutConcatTab = nn.ConcatTable()
            actSeqNN = nn.Sequential()
            actSeqNN:add(nn.Linear(lastHidNum, #classesActs))
            actSeqNN:add(nn.LogSoftMax())
            scoreSeqNN = nn.Sequential()
            scoreSeqNN:add(nn.Linear(lastHidNum, #classesScores))
            scoreSeqNN:add(nn.LogSoftMax())
            mulOutConcatTab:add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
            mulOutConcatTab:add(scoreSeqNN) -- {act, outcome(score)}

            self.model:add(mulOutConcatTab)
            self.model = nn.Sequencer(self.model)   -- This is interesting! This allows input to be a sequence of observations. We can also put FastLSTM into a Sequencer to substitue SeqLSTM, since SeqLSTM does not use RNN_dropout (Gal 16)
            ------------------------------------------------------------

        else
            print('Unknown model type')
            cmd:text()
            error()
        end

        -- params init
        local uapLinearLayers = self.model:findModules('nn.Linear')
        for l = 1, #uapLinearLayers do
            uapLinearLayers[l]:init('weight', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)}):init('bias', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)})    -- This bias initialization seems a little bit conflict with FastLSTM forget gate bias init. Maybe not, bcz it's on linear modules.
        end
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
    if opt.uppModel == 'lstm' then
        self.uaspPrlCriterion = nn.SequencerCriterion(self.uaspPrlCriterion)
    end

    self.trainEpoch = 1
    -- these matrices records the current confusion across classesActs and classesScores
    self.uapConfusion = optim.ConfusionMatrix(classesActs)
    self.uspConfusion = optim.ConfusionMatrix(classesScores)

    -- log results to files
    self.uaspTrainLogger = optim.Logger(paths.concat(opt.save, 'uaspTrain.log'))
    self.uaspTestLogger = optim.Logger(paths.concat(opt.save, 'uaspTest.log'))
    self.uaspTestLogger:setNames{'Epoch', 'Act Test acc.', 'Score Test acc.'}

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
            --            cutorch.manualSeed(opt.seed)
            --- set up cuda nn
            self.model = self.model:cuda()
            self.uapCriterion = self.uapCriterion:cuda()
            self.uspCriterion = self.uspCriterion:cuda()
            self.uaspPrlCriterion = self.uaspPrlCriterion:cuda()
            self.uaspPrlCriterion:add(self.uapCriterion)   -- action prediction loss function
            self.uaspPrlCriterion:add(self.uspCriterion)   -- score (outcome) prediction loss function
        else
            print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
            print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
            print('Falling back on CPU mode')
            opt.gpu = 0 -- overwrite user setting
        end
    end

    ----------------------------------------------------------------------
    --- Prepare data for lstm in training set
    ---
    self.rnnRealUserDataStates = {}
    self.rnnRealUserDataActs = {}
    self.rnnRealUserDataRewards = {}
    self.rnnRealUserDataStarts = {}
    self.rnnRealUserDataEnds = {}
    self.rnnRealUserDataPad = torch.Tensor(#self.ciUserSimulator.realUserDataStartLines):fill(0)    -- indicating whether data has padding at head (should be padded)
    if opt.uppModel == 'lstm' then
        local indSeqHead = 1
        local indSeqTail = opt.lstmHist
        local indUserSeq = 1    -- user id ptr. Use this to get the tail of each trajectory
        while indSeqTail <= #self.ciUserSimulator.realUserDataStates do
            if self.rnnRealUserDataPad[indUserSeq] < 1 then
                for padi = opt.lstmHist-1, 1, -1 do
                    self.rnnRealUserDataStates[#self.rnnRealUserDataStates + 1] = {}
                    self.rnnRealUserDataActs[#self.rnnRealUserDataActs + 1] = {}
                    self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards + 1] = {}
                    for i=1, padi do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i] = torch.Tensor(self.ciUserSimulator.userStateFeatureCnt):fill(0)
                        self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i] = self.ciUserSimulator.realUserDataActs[indSeqHead]  -- duplicate the 1st user action for padded states
                        self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards][i] = self.ciUserSimulator.realUserDataRewards[indSeqHead]
                    end
                    for i=1, opt.lstmHist-padi do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i+padi] = self.ciUserSimulator.realUserDataStates[indSeqHead+i-1]
                        self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i+padi] = self.ciUserSimulator.realUserDataActs[indSeqHead+i-1]
                        self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards][i+padi] = self.ciUserSimulator.realUserDataRewards[indSeqHead+i-1]
                    end
                    if padi == opt.lstmHist-1 then
                        self.rnnRealUserDataStarts[#self.rnnRealUserDataStarts+1] = #self.rnnRealUserDataStates     -- This is the start of a user's record -- This is duplicated. The value should be the same as realUserDataStartLines
                    end
                    if indSeqHead+(opt.lstmHist-padi)-1 == self.ciUserSimulator.realUserDataEndLines[indUserSeq] then
                        self.rnnRealUserDataPad[indUserSeq] = 1
                        break   -- if padding tail is going to outrange this user record's tail, break
                    end
                end
                self.rnnRealUserDataPad[indUserSeq] = 1
            else
                if indSeqTail <= self.ciUserSimulator.realUserDataEndLines[indUserSeq] then
                    self.rnnRealUserDataStates[#self.rnnRealUserDataStates + 1] = {}
                    self.rnnRealUserDataActs[#self.rnnRealUserDataActs + 1] = {}
                    self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards + 1] = {}
                    for i=1, opt.lstmHist do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i] = self.ciUserSimulator.realUserDataStates[indSeqHead+i-1]
                        self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i] = self.ciUserSimulator.realUserDataActs[indSeqHead+i-1]
                        self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards][i] = self.ciUserSimulator.realUserDataRewards[indSeqHead+i-1]
                    end
                    indSeqHead = indSeqHead + 1
                    indSeqTail = indSeqTail + 1
                else
                    self.rnnRealUserDataEnds[#self.rnnRealUserDataEnds+1] = #self.rnnRealUserDataStates     -- This is the end of a user's record
                    indUserSeq = indUserSeq + 1 -- next user's records
                    indSeqHead = self.ciUserSimulator.realUserDataStartLines[indUserSeq]
                    indSeqTail = indSeqHead + opt.lstmHist - 1
                end
            end
        end
        self.rnnRealUserDataEnds[#self.rnnRealUserDataEnds+1] = #self.rnnRealUserDataStates     -- Set the end of the last user's record
        -- There are in total 15509 sequences if histLen is 3. 14707 if histLen is 5. 15108 if histLen is 4. 15911 if histLen is 2.
    end

    ----------------------------------------------------------------------
    --- Prepare data for lstm in test/train_validation set
    ---
    self.rnnRealUserDataStatesTest = {}
    self.rnnRealUserDataActsTest = {}
    self.rnnRealUserDataRewardsTest = {}
    self.rnnRealUserDataStartsTest = {}
    self.rnnRealUserDataEndsTest = {}
    self.rnnRealUserDataPadTest = torch.Tensor(#self.ciUserSimulator.realUserDataStartLinesTest):fill(0)    -- indicating whether data has padding at head (should be padded)
    if self.opt.ciuTType == 'train' or self.opt.ciuTType == 'train_tr' then
        if opt.uppModel == 'lstm' then
            local indSeqHead = 1
            local indSeqTail = opt.lstmHist
            local indUserSeq = 1    -- user id ptr. Use this to get the tail of each trajectory
            while indSeqTail <= #self.ciUserSimulator.realUserDataStatesTest do
                if self.rnnRealUserDataPadTest[indUserSeq] < 1 then
                    for padi = opt.lstmHist-1, 1, -1 do
                        self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest + 1] = {}
                        self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest + 1] = {}
                        self.rnnRealUserDataRewardsTest[#self.rnnRealUserDataRewardsTest + 1] = {}
                        for i=1, padi do
                            self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest][i] = torch.Tensor(self.ciUserSimulator.userStateFeatureCnt):fill(0)
                            self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest][i] = self.ciUserSimulator.realUserDataActsTest[indSeqHead]  -- duplicate the 1st user action for padded states
                            self.rnnRealUserDataRewardsTest[#self.rnnRealUserDataRewardsTest][i] = self.ciUserSimulator.realUserDataRewardsTest[indSeqHead]
                        end
                        for i=1, opt.lstmHist-padi do
                            self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest][i+padi] = self.ciUserSimulator.realUserDataStatesTest[indSeqHead+i-1]
                            self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest][i+padi] = self.ciUserSimulator.realUserDataActsTest[indSeqHead+i-1]
                            self.rnnRealUserDataRewardsTest[#self.rnnRealUserDataRewardsTest][i+padi] = self.ciUserSimulator.realUserDataRewardsTest[indSeqHead+i-1]
                        end
                        if padi == opt.lstmHist-1 then
                            self.rnnRealUserDataStartsTest[#self.rnnRealUserDataStartsTest+1] = #self.rnnRealUserDataStatesTest     -- This is the start of a user's record
                        end
                        if indSeqHead+(opt.lstmHist-padi)-1 == self.ciUserSimulator.realUserDataEndLinesTest[indUserSeq] then
                            self.rnnRealUserDataPadTest[indUserSeq] = 1
                            break   -- if padding tail is going to outrange this user record's tail, break
                        end
                    end
                    self.rnnRealUserDataPadTest[indUserSeq] = 1
                else
                    if indSeqTail <= self.ciUserSimulator.realUserDataEndLinesTest[indUserSeq] then
                        self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest + 1] = {}
                        self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest + 1] = {}
                        self.rnnRealUserDataRewardsTest[#self.rnnRealUserDataRewardsTest + 1] = {}
                        for i=1, opt.lstmHist do
                            self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest][i] = self.ciUserSimulator.realUserDataStatesTest[indSeqHead+i-1]
                            self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest][i] = self.ciUserSimulator.realUserDataActsTest[indSeqHead+i-1]
                            self.rnnRealUserDataRewardsTest[#self.rnnRealUserDataRewardsTest][i] = self.ciUserSimulator.realUserDataRewardsTest[indSeqHead+i-1]
                        end
                        indSeqHead = indSeqHead + 1
                        indSeqTail = indSeqTail + 1
                    else
                        self.rnnRealUserDataEndsTest[#self.rnnRealUserDataEndsTest+1] = #self.rnnRealUserDataStatesTest     -- This is the end of a user's record
                        indUserSeq = indUserSeq + 1 -- next user's records
                        indSeqHead = self.ciUserSimulator.realUserDataStartLinesTest[indUserSeq]
                        indSeqTail = indSeqHead + opt.lstmHist - 1
                    end
                end
            end
            self.rnnRealUserDataEndsTest[#self.rnnRealUserDataEndsTest+1] = #self.rnnRealUserDataStatesTest     -- Set the end of the last user's record
            -- There are in total 15509 sequences if histLen is 3. 14707 if histLen is 5. 15108 if histLen is 4. 15911 if histLen is 2.
        end
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
    local closeToEnd
    local t = 1
    local lstmIter = 1  -- lstm iterate for each squence starts from this value
    local epochDone = false
    while not epochDone do
        if self.opt.uppModel ~= 'lstm' then
            -- create mini batch
            inputs = torch.Tensor(self.opt.batchSize, self.inputFeatureNum)
            targetsAct = torch.Tensor(self.opt.batchSize)
            targetsScore = torch.Tensor(self.opt.batchSize)
            closeToEnd = torch.Tensor(self.opt.batchSize):fill(0)
            local k = 1
            for i = t, math.min(t+self.opt.batchSize-1, #self.ciUserSimulator.realUserDataStates) do
                -- load new sample
                local input = self.ciUserSimulator.realUserDataStates[i]    -- :clone() -- if preprocess is called, clone is not needed, I believe
                -- need do preprocess for input features
                input = self.ciUserSimulator:preprocessUserStateData(input, self.opt.prepro)
                local singleTargetAct = self.ciUserSimulator.realUserDataActs[i]
                local singleTargetScore = self.ciUserSimulator.realUserDataRewards[i]
                inputs[k] = input
                targetsAct[k] = singleTargetAct
                targetsScore[k] = singleTargetScore
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
                    inputs[k] = self.ciUserSimulator:preprocessUserStateData(self.ciUserSimulator.realUserDataStates[randInd], self.opt.prepro)
                    targetsAct[k] = self.ciUserSimulator.realUserDataActs[randInd]
                    targetsScore[k] = self.ciUserSimulator.realUserDataRewards[randInd]
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

            if self.opt.gpu > 0 then
                inputs = inputs:cuda()
                targetsAct = targetsAct:cuda()
                targetsScore = targetsScore:cuda()
                closeToEnd = closeToEnd:cuda()
            end

            targetsActScore = {targetsAct, targetsScore}

        else
            -- lstm
            inputs = {}
            targetsAct = {}
            targetsScore = {}
            closeToEnd = torch.Tensor(self.opt.batchSize):fill(0)
            local k
            for j = 1, self.opt.lstmHist do
                inputs[j] = torch.Tensor(self.opt.batchSize, self.inputFeatureNum)
                targetsAct[j] = torch.Tensor(self.opt.batchSize)
                targetsScore[j] = torch.Tensor(self.opt.batchSize)
                k = 1
                for i = lstmIter, math.min(lstmIter+self.opt.batchSize-1, #self.rnnRealUserDataStates) do
                    local input = self.rnnRealUserDataStates[i][j]
                    input = self.ciUserSimulator:preprocessUserStateData(input, self.opt.prepro)
                    local singleTargetAct = self.rnnRealUserDataActs[i][j]
                    local singleTargetScore = self.rnnRealUserDataRewards[i][j]
                    inputs[j][k] = input
                    targetsAct[j][k] = singleTargetAct
                    targetsScore[j][k] = singleTargetScore
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
                        local input = self.rnnRealUserDataStates[randInd][j]
                        input = self.ciUserSimulator:preprocessUserStateData(input, self.opt.prepro)
                        local singleTargetAct = self.rnnRealUserDataActs[randInd][j]
                        local singleTargetScore = self.rnnRealUserDataRewards[randInd][j]
                        inputs[j][k] = input
                        targetsAct[j][k] = singleTargetAct
                        targetsScore[j][k] = singleTargetScore
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

            if self.opt.gpu > 0 then
                for _,v in pairs(inputs) do
                    v = v:cuda()
                end
                for _,v in pairs(targetsAct) do
                    v = v:cuda()
                end
                for _,v in pairs(targetsScore) do
                    v = v:cuda()
                end
                closeToEnd = closeToEnd:cuda()
            end

            for j = 1, self.opt.lstmHist do
                targetsActScore[j] = {}
                targetsActScore[j][1] = targetsAct[j]
                targetsActScore[j][2] = targetsScore[j]
            end

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
                    if self.opt.uppModel == 'lstm' then
                        outputs[self.opt.lstmHist][2][cl] = targetsActScore[self.opt.lstmHist][2][cl]
                    else
                        outputs[2][cl] = targetsActScore[2][cl]
                    end
                end
            end

            local f = self.uaspPrlCriterion:forward(outputs, targetsActScore) -- I made an experiment. The returned error value (f) from parallelCriterion is the sum of each criterion
            local df_do = self.uaspPrlCriterion:backward(outputs, targetsActScore)

            if self.opt.uppModel == 'lstm' then
                local subTot_f = 0
                for s=1, self.opt.lstmHist do
                    subTot_f = subTot_f + self.uapCriterion:forward(outputs[s][1], targetsActScore[s][1])
                end
                f = subTot_f + self.uspCriterion:forward(outputs[self.opt.lstmHist][2], targetsActScore[self.opt.lstmHist][2])

                for step=1, self.opt.lstmHist-1 do
                    df_do[step][2]:zero()   -- Zero df_do over Score prediction from time 1 to listHist-1
                end
            end

            -- If moe with shared lower layers, we merge the df_do before backprop
            if self.opt.uppModel == 'moe' then
                df_do = nn.JoinTable(-1):forward(df_do)  -- We assume 1st dim is batch index. Act pred is the 1st set of output, having dim of 15. Score dim 2.
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
            if self.opt.uppModel == 'lstm' then
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

            -- return f and df/dX
            return f, self.uaspDParam
        end

        self.model:training()
        if self.opt.uppModel == 'lstm' then
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
                learningRateDecay = 5e-7
            }
            optim.sgd(feval, self.uaspParam, sgdState)

            -- disp progress
            if self.opt.uppModel ~= 'lstm' then
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
            if self.opt.uppModel ~= 'lstm' then
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
            if self.opt.uppModel ~= 'lstm' then
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
    local filename = paths.concat(self.opt.save, 'uasp.t7')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    --    if paths.filep(filename) then
    --        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    --    end
    print('<trainer> saving ciunet to '..filename)
    torch.save(filename, self.model)

    if self.trainEpoch % 10 == 0 and self.opt.ciuTType == 'train' then
        filename = paths.concat(self.opt.save, string.format('%d', self.trainEpoch)..'_'..
                string.format('%.2f', self.uapConfusion.totalValid*100)..'_'..string.format('%.2f', self.uspConfusion.totalValid*100)..'uasp.t7')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('<trainer> saving periodly trained ciunet to '..filename)
        torch.save(filename, self.model)
    end

    if (self.opt.ciuTType == 'train' or self.opt.ciuTType == 'train_tr') and self.trainEpoch % self.opt.testOnTestFreq == 0 then
        local actScoreTestAccu = self:testActScorePredOnTestDetOneEpoch()
        print('<Act prediction accuracy at epoch '..string.format('%d', self.trainEpoch)..' on test set > '..string.format('%.2f%%', actScoreTestAccu[1]*100))
        print('<Score prediction accuracy at epoch '..string.format('%d', self.trainEpoch)..' on test set > '..string.format('%.2f%%', actScoreTestAccu[2]*100))
        self.uaspTestLogger:add{string.format('%d', self.trainEpoch), string.format('%.5f%%', actScoreTestAccu[1]*100), string.format('%.5f%%', actScoreTestAccu[2]*100)}
    end

    self.uapConfusion:zero()
    self.uspConfusion:zero()
    -- next epoch
    self.trainEpoch = self.trainEpoch + 1
end

-- evaluation functionon test/train_validation set
function CIUserActScorePredictor:testActScorePredOnTestDetOneEpoch()
    local tltCnt = 0
    local crcActCnt = 0
    local crcRewCnt = 0

    if self.opt.uppModel == 'lstm' then
        -- uSimShLayer == 1 and lstm model

        self.model:forget()
        self.model:evaluate()

        for i=1, #self.rnnRealUserDataStatesTest do
            local userState = self.rnnRealUserDataStatesTest[i]
            local userAct = self.rnnRealUserDataActsTest[i]
            local userRew = self.rnnRealUserDataRewardsTest[i]

            local tabState = {}
            for j=1, self.opt.lstmHist do
                local prepUserState = torch.Tensor(1, self.ciUserSimulator.userStateFeatureCnt)
                prepUserState[1] = self.ciUserSimulator:preprocessUserStateData(userState[j], self.opt.prepro)
                tabState[j] = prepUserState:clone()
            end

            local nll_acts = self.model:forward(tabState)   -- Here can be a problem for calling forward without considering GPU models. Not sure yet
            local lp, ain = torch.max(nll_acts[self.opt.lstmHist][1]:squeeze(), 1)     -- then 2nd [1] index is for action prediction from the shared act/score prediction outcome

            -- update action prediction confusion matrix
            if ain[1] == userAct[self.opt.lstmHist] then
                crcActCnt = crcActCnt + 1
            end

            if userAct[self.opt.lstmHist] == self.ciUserSimulator.CIFr.usrActInd_end then
                -- The predicted reward is the 2nd output of nll_acts in 2nd dim
                local lp, rin = torch.max(nll_acts[self.opt.lstmHist][2]:squeeze(), 1)
                if rin[1] == userRew[self.opt.lstmHist] then
                    crcRewCnt = crcRewCnt + 1
                end
            end

            tltCnt = tltCnt + 1
            self.model:forget()
        end

        -- return a table, with [1] being action pred accuracy, [2] being reward pred accuracy
        return {crcActCnt/tltCnt, crcRewCnt/#self.rnnRealUserDataEndsTest}
    else
        -- SharedLayer == 1, and not lstm models
        self.model:evaluate()
        for i=1, #self.ciUserSimulator.realUserDataStatesTest do
            local userState = self.ciUserSimulator:preprocessUserStateData(self.ciUserSimulator.realUserDataStatesTest[i], self.opt.prepro)
            local userAct = self.ciUserSimulator.realUserDataActsTest[i]
            local userRew = self.ciUserSimulator.realUserDataRewardsTest[i]

            local prepUserState = torch.Tensor(1, self.ciUserSimulator.userStateFeatureCnt)
            prepUserState[1] = userState:clone()

            local nll_acts = self.model:forward(prepUserState)  -- Here can be a problem for calling forward without considering GPU models. Not sure yet

            -- Here, if moe is used with shared lower layers, it is the problem that,
            -- due to limitation of MixtureTable module, we have to join tables together as
            -- a single tensor as output of the whole user action and score prediction model.
            -- So, to guarantee the compatability, we need split the tensor into two tables here,
            -- for act prediction and score prediction respectively.
            if self.opt.uppModel == 'moe' then
                nll_acts = nll_acts:split(self.ciUserSimulator.CIFr.usrActInd_end, 2)  -- We assume 1st dim is batch index. Act pred is the 1st set of output, having dim of 15. Score dim 2.
            end

            local lp, ain = torch.max(nll_acts[1][1]:squeeze(), 1)    -- The 1st [1] index means action prediction output from a table, 2nd [1] is batch index, which is not necessary

            -- update action prediction confusion matrix
            if ain[1] == userAct then
                crcActCnt = crcActCnt + 1
            end

            if userAct == self.ciUserSimulator.CIFr.usrActInd_end then
                -- The predicted reward is the 2nd output of nll_acts in 2nd dim
                local lp, rin = torch.max(nll_acts[2][1]:squeeze(), 1)
                if rin[1] == userRew then
                    crcRewCnt = crcRewCnt + 1
                end
            end

            tltCnt = tltCnt + 1
        end

        -- return a table, with [1] being action pred accuracy, [2] being reward pred accuracy
        return {crcActCnt/tltCnt, crcRewCnt/#self.ciUserSimulator.realUserDataEndLinesTest}
    end
end

return CIUserActScorePredictor


