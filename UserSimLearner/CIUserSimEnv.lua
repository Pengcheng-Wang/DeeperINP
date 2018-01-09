--
-- Created by IntelliJ IDEA.
-- User: pwang8
-- Date: 1/31/17
-- Time: 11:08 PM
-- This script aims at creating one script implementing the rlenvs APIs.
-- This script is modified based on UserBehaviorGenerator.lua
--
-- models separately. Also I'm thinking about maintaining only rnn type of running data structure, and then
-- use it to update one step structure or cnn type of data structure to feed into models. I've read to here. Dec 28, 2017.
--
-- How to use:
-- opt.ciunet == 'rlLoad'
-- opt.uSimShLayer: 1 for multi-task (player action, and outcome) models, 0 for bipartite player simulation model
-- opt.uSimScSoft: 1 for applying soft labels in player score (outcome) prediction, 0 for not use it
-- opt.uppModel: player action prediction model type
-- opt.uppModelUsp: player score prediction model type
-- opt.lstmHist: History length in input state representation used only in uap (user action predictor), not usp anymore
-- opt.lstmHistUsp: History length in input state representation used only in usp only
-- opt.rnnHdSizeL1: player action prediction RNN model layer size. We have not set up rnn layer size for score (outcome) predictor yet, because we always use layer size of 21 for all rnn/cnn models right now. This is used to construct dropout mask in this script
-- opt.rnnHdLyCnt: player action prediction RNN model hidden layer number. The same as above opt param. We don't set it up for score (outcome) predictor yet. This is used by dropout mask construction
-- opt.rnnHdSizeL1Usp: For usp model, score (outcome) predictor model hidden size
-- opt.rnnHdLyCntUsp: For usp model, score (outcome) predictor model hidden layer number
--

require 'torch'
require 'nn'
require 'nngraph'
local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'

local CIFileReader = require 'file_reader'
local CIUserSimulator = require 'UserSimulator'

local CIUserSimEnv = classic.class('CIUserSimEnv')

function CIUserSimEnv:_init(opt)

    -- Read CI trace and survey data files, and do validation
    local fr = CIFileReader()
    fr:evaluateTraceFile()
    fr:evaluateSurveyData()

    self.CIUSim = CIUserSimulator(fr, opt)
    self.CIUap = nil
    self.CIUsp = nil
    self.userActsPred = nil
    self.userScorePred = nil
    self.CIUasp = nil
    self.userActScorePred = nil
    self.opt = opt

    if opt.uSimShLayer < 0.5 then
        -- separate models for action and outcome (score) prediction
        local CIUserActsPredictor = require 'UserSimLearner/UserActsPredictor'
        local CIUserScorePredictor = self.opt.uSimScSoft == 0 and require 'UserSimLearner/UserScorePredictor' or require 'UserSimLearner/UserScoreSoftPredictor'
        self.CIUap = CIUserActsPredictor(self.CIUSim, opt)
        self.CIUsp = CIUserScorePredictor(self.CIUSim, opt)
        self.userActsPred = self.CIUap.model    -- set the reference of CIUserActsPredictor to the pre-loaded
        self.userScorePred = self.CIUsp.model

        -- Loading user simulation model from disks, and transfer them into cpu-based models
        self.userActsPred = cudnn.convert(self.userActsPred, nn)
        self.userActsPred = self.userActsPred:float()
        self.userScorePred = cudnn.convert(self.userScorePred, nn)
        self.userScorePred = self.userScorePred:float()

        self.userActsPred:evaluate()
        self.userScorePred:evaluate()
    else
        -- shared model for action and outcome (score) prediction
        local CIUserActScorePredictor = require 'UserSimLearner/UserActScorePredictor'
        self.CIUasp = CIUserActScorePredictor(self.CIUSim, opt)
        self.userActScorePred = self.CIUasp.model   -- set the reference of CIUserActScorePredictor to the pre-loaded

        -- Loading user simulation model from disks, and transfer them into cpu-based models
        self.userActScorePred = cudnn.convert(self.userActScorePred, nn)
        self.userActScorePred = self.userActScorePred:float()

        self.userActScorePred:evaluate()
    end

    --- Data generation related variables
    self.realDataStartsCnt = #self.CIUSim.realUserDataStartLines     -- count the number of real users
    self.rndStartInd = 1            -- Attention: self.CIUSim.realUserDataStartLines contains data only from training set.

    self.curRnnStatesRaw = nil
    self.curRnnUserAct = nil
    self.curOneStepStateRaw = nil
    self.curOneStepAct = nil

    -- We use the rnn form running data. Cnn form of data will be generated accordingly if needed
    local CIUp_model = self.CIUap
    if opt.uSimShLayer > 0.5 then
        CIUp_model = self.CIUasp
    end

    -- Both rnn and one-step form of running state will be kept, for the usage of different type of player sim models
    -- Cnn type data will not be kept. Cnn type of data can be directly formed from RNN type running state data
    self.curRnnStatesRaw = CIUp_model.rnnRealUserDataStates[CIUp_model.rnnRealUserDataStarts[self.rndStartInd]]     -- sample the 1st state
    self.curRnnUserAct = CIUp_model.rnnRealUserDataActs[CIUp_model.rnnRealUserDataStarts[self.rndStartInd]][self.opt.lstmHist] -- sample the 1st action (at last time step)

    self.curOneStepStateRaw = self.CIUSim.realUserDataStates[self.CIUSim.realUserDataStartLines[self.rndStartInd]]     -- sample the 1st state
    self.curOneStepAct = self.CIUSim.realUserDataActs[self.CIUSim.realUserDataStartLines[self.rndStartInd]] -- sample the 1st action


    self.adpTriggered = false
    self.adpType = 0  -- valid type value should range from 1 to 4
    self.rlStateRaw = torch.Tensor(1, 1, self.CIUSim.userStateFeatureCnt):fill(0)   -- this state should be 3d
    self.rlStatePrep = torch.Tensor(1, 1, self.CIUSim.userStateFeatureCnt):fill(0)   -- this state should be 3d
    self.rlStatePrepTypeInd = torch.Tensor(1, 1, self.CIUSim.userStateFeatureCnt + #self.CIUSim.CIFr.ciAdpActRanges):fill(0)   -- This representation contains 4 bits as adpType indicator
    self.nextSingleStepStateRaw = nil

    self.tabRnnStateRaw = {}   -- raw state value, used for updating future states according to current actions. This is state for user act/score prediction nn, not for rl
    for j=1, self.opt.lstmHist do
        local sinStepUserState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
        sinStepUserState[1] = self.curRnnStatesRaw[j]
        self.tabRnnStateRaw[j] = sinStepUserState:clone()
    end
    self.tabRnnStatePrep = {}
    self.curOneStepStatePrep = nil
    self.timeStepCnt = 1

end

------------------------------------------------
--- Create APIs following the format of rlenvs
--  All the following code is used by the RL script

function CIUserSimEnv:getStateSpec()
    -- Attention: for CI data used in rl training, the returned state representation contains 4-bit of adpType indicator (last 4 bit)
    return {'real', {1, 1, self.CIUSim.userStateFeatureCnt + #self.CIUSim.CIFr.ciAdpActRanges}, {0, 3}}    -- not sure about threshold of values, not guaranteed
end

-- 1 action required, of type 'int', of dimensionality 1, between 1 and 10
function CIUserSimEnv:getActionSpec()
    return {'int', 1, {1, 10}}
end

-- Return a table containing optional action choices lower/upper bounds
function CIUserSimEnv:getActBoundOfAdpType(adpT)
    return self.CIUSim.CIFr.ciAdpActRanges[adpT]
end

--- This function calculates the predicted user action according to current tabRnnStatePrep or curOneStepStatePrep value,
--- and sets self.curRnnUserAct (rnn, will also be used for cnn model data construction), or self.curOneStepAct (non-rnn, non-cnn)
function CIUserSimEnv:_calcUserAct()
    -- Pick an action using the action prediction model
    local nll_acts

    if self.opt.uSimShLayer < 0.5 then
        -- bipartite action, outcome (score) prediction models
        if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then   -- uppModel only indicating user action predictor model
            self.userActsPred:forget()
            --print(self.userActsPred)
            --os.exit()
            if self.opt.uppModelRNNDom > 0 then     -- if utilizing outside rnn dropout mask
                local simEnv_rnn_noise_i = {}
                local simEnv_rnn_noise_h = {}
                local simEnv_rnn_noise_o = {}
                TableSet.buildRNNDropoutMask(simEnv_rnn_noise_i, simEnv_rnn_noise_h, simEnv_rnn_noise_o, self.CIUSim.userStateFeatureCnt, self.opt.rnnHdSizeL1, self.opt.rnnHdLyCnt, 1, self.opt.lstmHist, self.opt.uppModelRNNDom)
                TableSet.sampleRNNDropoutMask(0, simEnv_rnn_noise_i, simEnv_rnn_noise_h, simEnv_rnn_noise_o, self.opt.rnnHdLyCnt, self.opt.lstmHist)
                local _tabRNNStatePrepWithDom = {}
                for j = 1, self.opt.lstmHist do
                    _tabRNNStatePrepWithDom[j] = {self.tabRnnStatePrep[j], simEnv_rnn_noise_i[j], simEnv_rnn_noise_h[j], simEnv_rnn_noise_o[j]}
                end
                nll_acts = self.userActsPred:forward(_tabRNNStatePrepWithDom)[self.opt.lstmHist]:squeeze() -- get act choice output for last time step
            else
                nll_acts = self.userActsPred:forward(self.tabRnnStatePrep)[self.opt.lstmHist]:squeeze() -- get act choice output for last time step
            end
        elseif string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then
            local _cnnStatePrep = torch.Tensor(1, self.opt.lstmHist, self.CIUSim.userStateFeatureCnt):zero()
            for hit=1, self.opt.lstmHist do
                _cnnStatePrep[1][hit] = self.tabRnnStatePrep[hit]:squeeze()
            end
            nll_acts = self.userActsPred:forward(_cnnStatePrep):squeeze()
        else
            -- non-rnn non-cnn action prediction model
            nll_acts = self.userActsPred:forward(self.curOneStepStatePrep):squeeze()
        end
    else
        -- unified (multi-task) action, outcome (score) prediction models
        if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' then
            self.userActScorePred:forget()
            if self.opt.uppModelRNNDom > 0 then
                local simEnv_rnn_noise_i = {}
                local simEnv_rnn_noise_h = {}
                local simEnv_rnn_noise_o = {}
                TableSet.buildRNNDropoutMask(simEnv_rnn_noise_i, simEnv_rnn_noise_h, simEnv_rnn_noise_o, self.CIUSim.userStateFeatureCnt, self.opt.rnnHdSizeL1, self.opt.rnnHdLyCnt, 1, self.opt.lstmHist, self.opt.uppModelRNNDom)
                TableSet.sampleRNNDropoutMask(0, simEnv_rnn_noise_i, simEnv_rnn_noise_h, simEnv_rnn_noise_o, self.opt.rnnHdLyCnt, self.opt.lstmHist)
                local _tabRNNStatePrepWithDom = {}
                for j = 1, self.opt.lstmHist do
                    _tabRNNStatePrepWithDom[j] = {self.tabRnnStatePrep[j], simEnv_rnn_noise_i[j], simEnv_rnn_noise_h[j], simEnv_rnn_noise_o[j]}
                end
                nll_acts = self.userActScorePred:forward(_tabRNNStatePrepWithDom)[self.opt.lstmHist]    -- get act choice output for last time step
            else
                nll_acts = self.userActScorePred:forward(self.tabRnnStatePrep)[self.opt.lstmHist]   -- get act choice output for last time step, act has index of 1
            end
            if string.sub(self.opt.uppModel, -3, -1) == 'moe' then
                nll_acts = nll_acts:split(self.CIUSim.CIFr.usrActInd_end, 2)
            end
            nll_acts = nll_acts[1]:squeeze()
        elseif string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then
            local _cnnStatePrep = torch.Tensor(1, self.opt.lstmHist, self.CIUSim.userStateFeatureCnt):zero()
            for hit=1, self.opt.lstmHist do
                _cnnStatePrep[1][hit] = self.tabRnnStatePrep[hit]:squeeze()
            end
            nll_acts = self.userActScorePred:forward(_cnnStatePrep)
            if string.sub(self.opt.uppModel, -3, -1) == 'moe' then
                nll_acts = nll_acts:split(self.CIUSim.CIFr.usrActInd_end, 2)
            end
            nll_acts = nll_acts[1][1]:squeeze()     -- the first index 1 is action prediction output (for multi-task output), second index 1 is batch index
        else
            -- non-rnn, non-cnn models
            nll_acts = self.userActScorePred:forward(self.curOneStepStatePrep)
            -- Here, if moe is used with shared lower layers, it is the problem that,
            -- due to limitation of MixtureTable module, we have to join tables together as
            -- a single tensor as output of the whole user action and score prediction model.
            -- So, to guarantee the compatability, we need split the tensor into two tables here,
            -- for act prediction and score prediction respectively.
            if self.opt.uppModel == 'moe' then
                nll_acts = nll_acts:split(self.CIUSim.CIFr.usrActInd_end, 2)  -- We assume 1st dim is batch index. Act pred is the 1st set of output, having dim of 15. Score dim 2.
            end
            nll_acts = nll_acts[1]:squeeze()
        end
    end

    --        print('Action choice likelihood Next time step:\n', torch.exp(nll_acts))
    local lpy   -- log likelihood value
    local lps   -- sorted index in desendent-order
    lpy, lps = torch.sort(nll_acts, 1, true)
    lpy = torch.exp(lpy)
    lpy = torch.cumsum(lpy)
    local actSampleLen = self.opt.actSmpLen
    lpy = torch.div(lpy, lpy[actSampleLen])
    local greedySmpThres = self.opt.actSmpEps

    self.curRnnUserAct = lps[1]  -- the action result given by the action predictor
    self.curOneStepAct = lps[1]  -- if it is non-lstm model
    if torch.uniform() > greedySmpThres then
        -- sample according to classifier output
        local rndActPick = torch.uniform()
        for i=1, actSampleLen do
            if rndActPick <= lpy[i] then
                self.curRnnUserAct = lps[i]
                self.curOneStepAct = lps[i]
                break
            end
        end
    end

    ---- The following code controls generation of ending user action
    --     if user action sequence is too long, we can manually add this end
    --     action to terminate the sequence, at the same time influence the
    --     action distribution a little. This can be a safe design, but does
    --     not have to be necessary.
    if self.timeStepCnt >= self.opt.termActSmgLen then
        if torch.uniform() < self.opt.termActSmgEps then
            self.curRnnUserAct = self.CIUSim.CIFr.usrActInd_end
            self.curOneStepAct = self.CIUSim.CIFr.usrActInd_end
        end
    end

    return self.curRnnUserAct
end


function CIUserSimEnv:_updateRnnStatePrep()
    -- states after preprocessing
    for j=1, self.opt.lstmHist do
        local prepSinStepState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
        prepSinStepState[1] = self.CIUSim:preprocessUserStateData(self.tabRnnStateRaw[j][1], self.opt.prepro)
        self.tabRnnStatePrep[j] = prepSinStepState:clone()
    end
end


function CIUserSimEnv:_updateOneStepStatePrep()
    -- states after preprocessing
    self.curOneStepStatePrep = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
    self.curOneStepStatePrep[1] = self.CIUSim:preprocessUserStateData(self.curOneStepStateRaw, self.opt.prepro)
end


function CIUserSimEnv:_updateRLStatePrepTypeInd(endingType)
    self.rlStatePrepTypeInd:zero()
    for i=1, self.rlStatePrep:size(3) do
        self.rlStatePrepTypeInd[1][1][i] = self.rlStatePrep[1][1][i]
    end
    if not endingType then
        self.rlStatePrepTypeInd[1][1][self.rlStatePrep:size(3) + self.adpType] = 1
    end
end


function CIUserSimEnv:start()
    local valid = false

    while not valid do
        --- randomly select one human user's record whose 1st action cannot be ending action
        if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' or string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then      -- if action predictor type is rnn or cnn
            local CIUp_model = self.CIUap
            if self.opt.uSimShLayer > 0.5 then
                CIUp_model = self.CIUasp
            end

            repeat
                self.rndStartInd = torch.random(1, self.realDataStartsCnt)
            until CIUp_model.rnnRealUserDataActs[CIUp_model.rnnRealUserDataStarts[self.rndStartInd]][self.opt.lstmHist] ~= self.CIUSim.CIFr.usrActInd_end

            --- Get this user's state record at the 1st time stpe. This process means we sample
            --  user's 1st action and survey data from human user's records. Then we use our prediction
            --  model to estimate user's future ations.
            self.curRnnStatesRaw = CIUp_model.rnnRealUserDataStates[CIUp_model.rnnRealUserDataStarts[self.rndStartInd]]     -- sample the 1st state
            self.curRnnUserAct = CIUp_model.rnnRealUserDataActs[CIUp_model.rnnRealUserDataStarts[self.rndStartInd]][self.opt.lstmHist] -- sample the 1st action (at last time step)

            self.tabRnnStateRaw = {}   -- raw state value, used for updating future states according to current actions. This is state for user act/score prediction nn, not for rl
            for j=1, self.opt.lstmHist do
                self.sinStepUserState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
                self.sinStepUserState[1] = self.curRnnStatesRaw[j]
                self.tabRnnStateRaw[j] = self.sinStepUserState:clone()
            end

            -- When user ap/sp state and action were given, check if adaptation could be triggered
            -- valid type value should range from 1 to 4
            self.adpTriggered, self.adpType = self.CIUSim:isAdpTriggered(self.tabRnnStateRaw[self.opt.lstmHist], self.curRnnUserAct)
        else
            -- non-rnn, non-cnn models
            repeat
                self.rndStartInd = torch.random(1, self.realDataStartsCnt)
            until self.CIUSim.realUserDataActs[self.CIUSim.realUserDataStartLines[self.rndStartInd]] ~= self.CIUSim.CIFr.usrActInd_end

            --- Get this user's state record at the 1st time step. This process means we sample
            --  user's 1st action and survey data from human user's records. Then we use our prediction
            --  model to estimate user's future ations.
            self.curOneStepStateRaw = self.CIUSim.realUserDataStates[self.CIUSim.realUserDataStartLines[self.rndStartInd]]     -- sample the 1st state
            self.curOneStepAct = self.CIUSim.realUserDataActs[self.CIUSim.realUserDataStartLines[self.rndStartInd]] -- sample the 1st action

            -- When user ap/sp state and action were given, check if adaptation could be triggered
            -- valid type value should range from 1 to 4
            self.adpTriggered, self.adpType = self.CIUSim:isAdpTriggered(self.curOneStepStateRaw, self.curOneStepAct)
        end

        self.timeStepCnt = 1
        self.tabRnnStatePrep = {}

        if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' or string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then
            while not self.adpTriggered and self.curRnnUserAct ~= self.CIUSim.CIFr.usrActInd_end do
                -- apply user's action onto raw state representation
                -- This is the state representation for next single time step
                self.nextSingleStepStateRaw = self.tabRnnStateRaw[self.opt.lstmHist]:clone()
                self.CIUSim:applyUserActOnState(self.nextSingleStepStateRaw, self.curRnnUserAct)
                -- print('--- Next single step rnn state raw', self.nextSingleStepStateRaw)

                -- reconstruct rnn state table for next time step
                for j=1, self.opt.lstmHist-1 do
                    self.tabRnnStateRaw[j] = self.tabRnnStateRaw[j+1]:clone()
                end
                self.tabRnnStateRaw[self.opt.lstmHist] = self.nextSingleStepStateRaw:clone()

                self.timeStepCnt = self.timeStepCnt + 1
                self:_updateRnnStatePrep()
                self:_calcUserAct()

                --            print(self.timeStepCnt, 'time step state:') for k,v in ipairs(self.tabRnnStateRaw) do print(k,v) end
                --            print(self.timeStepCnt, 'time step act:', self.curRnnUserAct)

                -- When user ap/sp state and action were given, check if adaptation could be triggered
                self.adpTriggered, self.adpType = self.CIUSim:isAdpTriggered(self.tabRnnStateRaw[self.opt.lstmHist], self.curRnnUserAct)

            end -- end of while
        else
            -- non-rnn, non-cnn models
            while not self.adpTriggered and self.curOneStepAct ~= self.CIUSim.CIFr.usrActInd_end do
                -- apply user's action onto raw state representation
                -- This is the state representation for next single time step
                self.CIUSim:applyUserActOnState(self.curOneStepStateRaw, self.curOneStepAct)
                -- print('--- Next single step rnn state raw', self.curOneStepStateRaw)

                self.timeStepCnt = self.timeStepCnt + 1
                self:_updateOneStepStatePrep()
                self:_calcUserAct()

                --            print(self.timeStepCnt, 'time step state:', self.curOneStepStateRaw)
                --            print(self.timeStepCnt, 'time step act:', self.curOneStepAct)

                -- When user ap/sp state and action were given, check if adaptation could be triggered
                self.adpTriggered, self.adpType = self.CIUSim:isAdpTriggered(self.curOneStepStateRaw, self.curOneStepAct)

            end -- end of while
        end

        self.rlStateRaw = torch.Tensor(1, 1, self.CIUSim.userStateFeatureCnt):fill(0)   -- this state should be 3d
        self.rlStatePrep = torch.Tensor(1, 1, self.CIUSim.userStateFeatureCnt):fill(0)   -- this state should be 3d

        -- Attention: we guarantee that the ending user action will not trigger adaptation
        if self.adpTriggered then
            --            print('--- Adp triggered')

            if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' or string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then
                self.rlStateRaw[1][1] = self.tabRnnStateRaw[self.opt.lstmHist][1] -- copy the last time step RAW state representation. Clone() is not needed.

                -- Need to add the user action's effect on rl state
                self.CIUSim:applyUserActOnState(self.rlStateRaw, self.curRnnUserAct)
            else
                -- non-rnn, non-cnn models
                self.rlStateRaw[1][1] = self.curOneStepStateRaw -- copy the last time step RAW state representation. Clone() is not needed.

                -- Need to add the user action's effect on rl state
                self.CIUSim:applyUserActOnState(self.rlStateRaw, self.curOneStepAct)
            end

            -- Need to add the user action's effect on rl state
            self.rlStatePrep[1][1] = self.CIUSim:preprocessUserStateData(self.rlStateRaw[1][1], self.opt.prepro)   -- do preprocessing before sending back to RL
            --            print('--- After apply user act, rl state:', self.rlStateRaw[1][1])
            --            print('--- Prep rl state', self.rlStatePrep[1][1])
            -- Should get action choice from the RL agent here

            valid = true    -- not necessary
            --            return self.rlStatePrep, self.adpType
            self:_updateRLStatePrepTypeInd()
            -- It seems to be necessary to return a clone of rlStatePrepTypeInd, because in async models, multiple threads may use this object at the same time
            return self.rlStatePrepTypeInd:clone(), self.adpType

        else    -- self.curRnnUserAct == self.CIUSim.CIFr.usrActInd_end
            --            print('Regenerate user behavior trajectory from start!')
            valid = false   -- not necessary
        end

    end
end


function CIUserSimEnv:step(adpAct)
    assert(adpAct >= self.CIUSim.CIFr.ciAdpActRanges[self.adpType][1] and adpAct <= self.CIUSim.CIFr.ciAdpActRanges[self.adpType][2])

    if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' or string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then

        self.nextSingleStepStateRaw = self.tabRnnStateRaw[self.opt.lstmHist]:clone()
        self.CIUSim:applyAdpActOnState(self.nextSingleStepStateRaw, self.adpType, adpAct)

        repeat

            self.CIUSim:applyUserActOnState(self.nextSingleStepStateRaw, self.curRnnUserAct)
            -- reconstruct rnn state table for next time step
            for j=1, self.opt.lstmHist-1 do
                self.tabRnnStateRaw[j] = self.tabRnnStateRaw[j+1]:clone()
            end
            self.tabRnnStateRaw[self.opt.lstmHist] = self.nextSingleStepStateRaw:clone()

            self.timeStepCnt = self.timeStepCnt + 1
            self:_updateRnnStatePrep()
            self:_calcUserAct()

            --        print(self.timeStepCnt, 'time step state:') for k,v in ipairs(self.tabRnnStateRaw) do print(k,v) end
            --        print(self.timeStepCnt, 'time step act:', self.curRnnUserAct)

            -- When user ap/sp state and action were given, check if adaptation could be triggered
            self.adpTriggered, self.adpType = self.CIUSim:isAdpTriggered(self.tabRnnStateRaw[self.opt.lstmHist], self.curRnnUserAct)

            self.nextSingleStepStateRaw = self.tabRnnStateRaw[self.opt.lstmHist]:clone()

        until self.adpTriggered or self.curRnnUserAct == self.CIUSim.CIFr.usrActInd_end

    else
        -- non-lstm models
        self.CIUSim:applyAdpActOnState(self.curOneStepStateRaw, self.adpType, adpAct)

        repeat

            self.CIUSim:applyUserActOnState(self.curOneStepStateRaw, self.curOneStepAct)

            self.timeStepCnt = self.timeStepCnt + 1
            self:_updateOneStepStatePrep()
            self:_calcUserAct()

            --            print(self.timeStepCnt, 'time step state:', self.curOneStepStateRaw)
            --            print(self.timeStepCnt, 'time step act:', self.curOneStepAct)

            -- When user ap/sp state and action were given, check if adaptation could be triggered
            self.adpTriggered, self.adpType = self.CIUSim:isAdpTriggered(self.curOneStepStateRaw, self.curOneStepAct)

        until self.adpTriggered or self.curOneStepAct == self.CIUSim.CIFr.usrActInd_end

    end

    -- Attention: we guarantee that the ending user action will not trigger adaptation
    if self.adpTriggered then
        --        print('--- Adp triggered')
        if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' or string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then
            self.rlStateRaw[1][1] = self.tabRnnStateRaw[self.opt.lstmHist][1] -- copy the last time step RAW state representation. Clone() is not needed.

            -- Need to add the user action's effect on rl state
            self.CIUSim:applyUserActOnState(self.rlStateRaw, self.curRnnUserAct)
        else
            -- non-lstm models
            self.rlStateRaw[1][1] = self.curOneStepStateRaw -- copy the last time step RAW state representation. Clone() is not needed.

            -- Need to add the user action's effect on rl state
            self.CIUSim:applyUserActOnState(self.rlStateRaw, self.curOneStepAct)
        end

        -- Need to add the user action's effect on rl state
        self.rlStatePrep[1][1] = self.CIUSim:preprocessUserStateData(self.rlStateRaw[1][1], self.opt.prepro)   -- do preprocessing before sending back to RL
        --            print('--- After apply user act, rl state:', self.rlStateRaw[1][1])
        --            print('--- Prep rl state', self.rlStatePrep[1][1])
        -- Should get action choice from the RL agent here

        self:_updateRLStatePrepTypeInd()
        return 0, self.rlStatePrepTypeInd:clone(), false, self.adpType

    else    -- self.curRnnUserAct == self.CIUSim.CIFr.usrActInd_end

        if string.sub(self.opt.uppModel, 1, 4) == 'rnn_' or string.sub(self.opt.uppModel, 1, 4) == 'cnn_' then
            self.rlStateRaw[1][1] = self.tabRnnStateRaw[self.opt.lstmHist][1] -- copy the last time step RAW state representation. Clone() is not needed.
        else
            -- non-lstm models
            self.rlStateRaw[1][1] = self.curOneStepStateRaw -- copy the last time step RAW state representation. Clone() is not needed.
        end
        -- Does not need to apply an ending user action. It will not change state representation.
        self.rlStatePrep[1][1] = self.CIUSim:preprocessUserStateData(self.rlStateRaw[1][1], self.opt.prepro)   -- do preprocessing before sending back to RL

        --- Score (outcome) prediction
        local nll_rewards
        local nll_rwd

        if self.opt.uSimShLayer < 0.5 then
            -- Bipartite action, outcome (score) prediction models
            if string.sub(self.opt.uppModelUsp, 1, 4) == 'rnn_' then   -- uppModelUsp indicating user score predictor model
                self:_updateRnnStatePrep()
                self.userScorePred:forget()
                assert(self.opt.lstmHist >= self.opt.lstmHistUsp, 'Action predictor history length should >= score (outcome) predictor history length')
                local _rnnScorePredInputPrep
                if self.opt.lstmHist > self.opt.lstmHistUsp then
                    _rnnScorePredInputPrep = {}
                    for itr=1, self.opt.lstmHistUsp do
                        _rnnScorePredInputPrep[itr] = self.tabRnnStatePrep[self.opt.lstmHist - self.opt.lstmHistUsp + itr]
                    end
                else
                    _rnnScorePredInputPrep = self.tabRnnStatePrep
                end

                if self.opt.uppModelRNNDomUsp > 0 then     -- if utilizing outside rnn dropout mask for user score predictor
                    local simEnv_rnn_noise_i = {}
                    local simEnv_rnn_noise_h = {}
                    local simEnv_rnn_noise_o = {}
                    TableSet.buildRNNDropoutMask(simEnv_rnn_noise_i, simEnv_rnn_noise_h, simEnv_rnn_noise_o, self.CIUSim.userStateFeatureCnt, self.opt.rnnHdSizeL1Usp, self.opt.rnnHdLyCntUsp, 1, self.opt.lstmHistUsp, self.opt.uppModelRNNDomUsp)
                    TableSet.sampleRNNDropoutMask(0, simEnv_rnn_noise_i, simEnv_rnn_noise_h, simEnv_rnn_noise_o, self.opt.rnnHdLyCntUsp, self.opt.lstmHistUsp)
                    local _tabRNNStatePrepWithDom = {}
                    for j = 1, self.opt.lstmHistUsp do
                        _tabRNNStatePrepWithDom[j] = {_rnnScorePredInputPrep[j], simEnv_rnn_noise_i[j], simEnv_rnn_noise_h[j], simEnv_rnn_noise_o[j]}
                    end
                    nll_rewards = self.userScorePred:forward(_tabRNNStatePrepWithDom)
                else
                    nll_rewards = self.userScorePred:forward(_rnnScorePredInputPrep)
                end

                -- If ScoreSoftPredictor is utilized
                if self.opt.uSimScSoft > 0 then
                    if string.sub(self.opt.uppModelUsp, -3, -1) == 'moe' then
                        nll_rewards[self.opt.lstmHistUsp] = nll_rewards[self.opt.lstmHistUsp]:split(2, 2)  -- We assume 1st dim is batch index. Score classification is the 1st set of output, having 2 outputs. Score regression is the 2nd set of output.
                    end
                    nll_rwd = nll_rewards[self.opt.lstmHistUsp][1]:squeeze()  -- nll_rewards[self.opt.lstmHistUsp] is a table of 2 items. nll_rewards[self.opt.lstmHistUsp][1] is a 2-dim tensor, with 1st dim being batch index (only 1 in a batch). split(2, 1) means split along 1st dim, and 1st set contains 2 output units
                else
                    -- If ScorePredictor is used (NOT ScoreSoftPredictor)
                    nll_rwd = nll_rewards[self.opt.lstmHistUsp]:squeeze()
                end
            elseif string.sub(self.opt.uppModelUsp, 1, 4) == 'cnn_' then
                self:_updateRnnStatePrep()
                assert(self.opt.lstmHist >= self.opt.lstmHistUsp, 'Action predictor history length should >= score (outcome) predictor history length')
                local _cnnScorePredInputPrep = torch.Tensor(1, self.opt.lstmHistUsp, self.CIUSim.userStateFeatureCnt):zero()
                for itr=1, self.opt.lstmHistUsp do
                    _cnnScorePredInputPrep[1][itr] = self.tabRnnStatePrep[self.opt.lstmHist - self.opt.lstmHistUsp + itr][1]
                end
                nll_rewards = self.userScorePred:forward(_cnnScorePredInputPrep)

                -- If ScoreSoftPredictor is utilized
                if self.opt.uSimScSoft > 0 then
                    if string.sub(self.opt.uppModelUsp, -3, -1) == 'moe' then
                        nll_rewards = nll_rewards:split(2, 2)  -- We assume 1st dim is batch index. Score classification is the 1st set of output, having 2 outputs. Score regression is the 2nd set of output.
                    end
                    nll_rwd = nll_rewards[1]:squeeze()
                else
                    -- If ScorePredictor is used (NOT ScoreSoftPredictor)
                    nll_rwd = nll_rewards:squeeze()
                end
            else
                -- non-rnn, non-cnn models
                self:_updateOneStepStatePrep()
                nll_rewards = self.userScorePred:forward(self.curOneStepStatePrep)
                nll_rwd = nll_rewards:squeeze()
            end
        else
            -- self.opt.uSimShLayer == 1
            -- Unified action, outcome (score) prediction models with shared layer structure
            if string.sub(self.opt.uppModelUsp, 1, 4) == 'rnn_' then   -- uppModelUsp indicating user score predictor model
                self:_updateRnnStatePrep()
                self.userActScorePred:forget()
                assert(self.opt.lstmHist >= self.opt.lstmHistUsp, 'Action predictor history length should >= score (outcome) predictor history length')
                local _rnnScorePredInputPrep
                if self.opt.lstmHist > self.opt.lstmHistUsp then
                    _rnnScorePredInputPrep = {}
                    for itr=1, self.opt.lstmHistUsp do
                        _rnnScorePredInputPrep[itr] = self.tabRnnStatePrep[self.opt.lstmHist - self.opt.lstmHistUsp + itr]
                    end
                else
                    _rnnScorePredInputPrep = self.tabRnnStatePrep
                end

                if self.opt.uppModelRNNDomUsp > 0 then     -- if utilizing outside rnn dropout mask
                    local simEnv_rnn_noise_i = {}
                    local simEnv_rnn_noise_h = {}
                    local simEnv_rnn_noise_o = {}
                    TableSet.buildRNNDropoutMask(simEnv_rnn_noise_i, simEnv_rnn_noise_h, simEnv_rnn_noise_o, self.CIUSim.userStateFeatureCnt, self.opt.rnnHdSizeL1Usp, self.opt.rnnHdLyCntUsp, 1, self.opt.lstmHistUsp, self.opt.uppModelRNNDomUsp)     -- even though multi-task action score prediction models have both action and score predictions, we still use two different sets of opt params in case that the multi-task model will only be used for score prediction
                    TableSet.sampleRNNDropoutMask(0, simEnv_rnn_noise_i, simEnv_rnn_noise_h, simEnv_rnn_noise_o, self.opt.rnnHdLyCntUsp, self.opt.lstmHistUsp)
                    local _tabRNNStatePrepWithDom = {}
                    for j = 1, self.opt.lstmHistUsp do
                        _tabRNNStatePrepWithDom[j] = {_rnnScorePredInputPrep[j], simEnv_rnn_noise_i[j], simEnv_rnn_noise_h[j], simEnv_rnn_noise_o[j]}
                    end
                    nll_rewards = self.userActScorePred:forward(_tabRNNStatePrepWithDom)
                else
                    nll_rewards = self.userActScorePred:forward(_rnnScorePredInputPrep)
                end

                if string.sub(self.opt.uppModelUsp, -3, -1) == 'moe' then
                    nll_rewards[self.opt.lstmHistUsp] = nll_rewards[self.opt.lstmHistUsp]:split(self.CIUSim.CIFr.usrActInd_end, 2)  -- We assume 1st dim is batch index. Score classification is the 1st set of output, having 2 outputs. Score regression is the 2nd set of output.
                end
                nll_rwd = nll_rewards[self.opt.lstmHistUsp][2]:squeeze()
            elseif string.sub(self.opt.uppModelUsp, 1, 4) == 'cnn_' then
                self:_updateRnnStatePrep()
                assert(self.opt.lstmHist >= self.opt.lstmHistUsp, 'Action predictor history length should >= score (outcome) predictor history length')
                local _cnnScorePredInputPrep = torch.Tensor(1, self.opt.lstmHistUsp, self.CIUSim.userStateFeatureCnt):zero()
                for itr=1, self.opt.lstmHistUsp do
                    _cnnScorePredInputPrep[1][itr] = self.tabRnnStatePrep[self.opt.lstmHist - self.opt.lstmHistUsp + itr][1]
                end
                nll_rewards = self.userActScorePred:forward(_cnnScorePredInputPrep)

                if string.sub(self.opt.uppModelUsp, -3, -1) == 'moe' then
                    nll_rewards = nll_rewards:split(self.CIUSim.CIFr.usrActInd_end, 2)  -- We assume 1st dim is batch index. Score classification is the 1st set of output, having 2 outputs. Score regression is the 2nd set of output.
                end
                nll_rwd = nll_rewards[2]:squeeze()
            else
                -- non-lstm models
                self:_updateOneStepStatePrep()
                -- non-lstm models
                nll_rewards = self.userActScorePred:forward(self.curOneStepStatePrep)
                -- Here, if moe is used with shared lower layers, it is the problem that,
                -- due to limitation of MixtureTable module, we have to join tables together as
                -- a single tensor as output of the whole user action and score prediction model.
                -- So, to guarantee the compatability, we need split the tensor into two tables here,
                -- for act prediction and score prediction respectively.
                if string.sub(self.opt.uppModelUsp, -3, -1) == 'moe' then
                    nll_rewards = nll_rewards:split(self.CIUSim.CIFr.usrActInd_end, 2)  -- We assume 1st dim is batch index. Act pred is the 1st set of output, having dim of 15. Score dim 2.
                end
                nll_rwd = nll_rewards[2]:squeeze()
            end
        end


        local lpy   -- log likelihood value
        local lps   -- sorted index in desendent-order
        lpy, lps = torch.sort(nll_rwd, 1, true)
        lpy = torch.exp(lpy)
        lpy = torch.cumsum(lpy)
        local rwdSampleLen = 2  -- two types of rewards assumption in CI
        --        lpy = torch.div(lpy, lpy[rwdSampleLen])
        local greedySmpThres = self.opt.rwdSmpEps

        local scoreType = lps[1]  -- the action result given by the action predictor
        if torch.uniform() > greedySmpThres then
            -- sample according to classifier output
            local rndRwdPick = torch.uniform()
            for i=1, rwdSampleLen do
                if rndRwdPick <= lpy[i] then
                    scoreType = lps[i]
                    break
                end
            end
        end

        local score = 1
        if scoreType == 2 then score = self.opt.ciGroup2rwd end
        self:_updateRLStatePrepTypeInd(true)    -- pass true as param to indicate ending act is reached
        return score, self.rlStatePrepTypeInd:clone(), true, 0
    end

end

--- Set up the trianing mode for this rl environment
function CIUserSimEnv:training()
end

--- Set up the evaluate mode for this rl environment
function CIUserSimEnv:evaluate()
end

--- Returns (RGB) display of screen, Fake function for CIUserSimEnv
function CIUserSimEnv:getDisplay()
    return torch.repeatTensor(torch.div(self.rlStateRaw, 50), 3, 1, 1)
end

--- RGB screen of size self.size x self.size
function CIUserSimEnv:getDisplaySpec()
    return {'real', {3, 1, self.CIUSim.userStateFeatureCnt}, {0, 1}}
end

return CIUserSimEnv
