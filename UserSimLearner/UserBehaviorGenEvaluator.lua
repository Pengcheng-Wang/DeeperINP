--
-- User: pwang8
-- Date: 2/5/17
-- Time: 4:59 PM
-- This file is used for testing the performance of user action/score predictors' performance on test set.
-- And this script is modified from a pervious version of UserBehaviorGenerator.lua
--

require 'torch'
require 'nn'
local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'  -- not used right now

local CIUserBehaviorGenEvaluator = classic.class('UserBehaviorGenEvaluator')

function CIUserBehaviorGenEvaluator:_init(CIUserSimulator, CIUserActsPred, CIUserScorePred, CIUserActScorePred, opt)

    local tltCnt = 0
    local crcActCnt = 0
    local crcRewCnt = 0
    local userInd = 1
    local earlyTotAct = torch.Tensor(opt.lstmHist+81):fill(1e-6)
    local earlyCrcAct = torch.Tensor(opt.lstmHist+81):fill(0)
    local firstActDist = torch.Tensor(CIUserSimulator.CIFr.usrActInd_end):fill(0)   -- 15 user actions

    -- Confusion matrix for action prediction (15 class)
    local actPredTP = torch.Tensor(CIUserSimulator.CIFr.usrActInd_end):fill(1e-3)
    local actPredFP = torch.Tensor(CIUserSimulator.CIFr.usrActInd_end):fill(1e-3)
    local actPredFN = torch.Tensor(CIUserSimulator.CIFr.usrActInd_end):fill(1e-3)
    -- Confusion matrix for positive score (outcome) prediction (binary)
    local scorePredTP = torch.Tensor(2):fill(1e-3)
    local scorePredFP = torch.Tensor(2):fill(1e-3)
    local scorePredFN = torch.Tensor(2):fill(1e-3)

    local countScope=0  -- This param is used to calculate action distribution at countScope time step

    self:userActionDistStats(CIUserSimulator)

    if opt.uSimShLayer < 1 then

        -- Bipartitle Act/Score prediction model
        -- The user action/score predictors for evaluation should be already trained, and loaded from files
        -- Also, the CIUserSimulator, CIUserActsPred, CIUserScorePred should be initialized using
        -- the test set.
        print('User Act Score Bipartitle model: #', paths.concat(opt.ubgDir , opt.uapFile),
            ', ', paths.concat(opt.ubgDir , opt.uspFile))
        self.userActsPred = torch.load(paths.concat(opt.ubgDir , opt.uapFile))
        self.userScorePred = torch.load(paths.concat(opt.ubgDir , opt.uspFile))
        self.userActsPred:evaluate()
        self.userScorePred:evaluate()
        self.CIUSim = CIUserSimulator
        self.CIUap = CIUserActsPred
        self.CIUsp = CIUserScorePred
        self.opt = opt

        if opt.uppModel == 'lstm' then
            -- uSimShLayer == 0 and lstm model
--            self._actionDistributionCalc(CIUserSimulator, countScope)

            self.userActsPred:forget()
            self.userScorePred:forget()

            for i=1, #CIUserActsPred.rnnRealUserDataStates do
                local userState = CIUserActsPred.rnnRealUserDataStates[i]
                local userAct = CIUserActsPred.rnnRealUserDataActs[i]
                local userRew = CIUserScorePred.rnnRealUserDataRewards[i]

                local tabState = {}
                for j=1, opt.lstmHist do
                    local prepUserState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
                    prepUserState[1] = CIUserSimulator:preprocessUserStateData(userState[j], opt.prepro)
                    tabState[j] = prepUserState:clone()
                end

                local nll_acts = self.userActsPred:forward(tabState)
                lp, ain = torch.max(nll_acts[opt.lstmHist]:squeeze(), 1)
                if i == CIUserActsPred.rnnRealUserDataStarts[userInd]+countScope then firstActDist[ain[1]] = firstActDist[ain[1]]+1 end   -- check act dist at each x-th time step

                -- update action prediction confusion matrix
                if ain[1] == userAct[opt.lstmHist] then
                    actPredTP[ain[1]] = actPredTP[ain[1]] + 1
                else
                    actPredFP[ain[1]] = actPredFP[ain[1]] + 1
                    actPredFN[userAct[opt.lstmHist]] = actPredFN[userAct[opt.lstmHist]] + 1
                end

                --            if ain[1] == userAct[opt.lstmHist] then crcActCnt = crcActCnt + 1 end
                lpy, lps = torch.sort(nll_acts[opt.lstmHist]:squeeze(), 1, true)
                local crtExt = false
                local smpLen = self.opt.actEvaScp    -- This is the range of selected actions to check for correct prediction
                for l=1, smpLen do
                    if lps[l] == userAct[opt.lstmHist] then
                        crcActCnt = crcActCnt + 1
                        crtExt = true
                        break
                    end
                end

                local indDiff = i - CIUserActsPred.rnnRealUserDataStarts[userInd]
                if indDiff >= 0 and indDiff <= opt.lstmHist+80 and i <= CIUserActsPred.rnnRealUserDataEnds[userInd] then
                    if crtExt then
                        earlyCrcAct[indDiff+1] = earlyCrcAct[indDiff+1] + 1 -- This is at the indDiff's time step, what is prob of correct act prediction
                    end
                    earlyTotAct[indDiff+1] = earlyTotAct[indDiff+1] + 1
                end
                if i == CIUserActsPred.rnnRealUserDataEnds[userInd] then
                    userInd = userInd+1
                end

                if userAct[opt.lstmHist] == CIUserSimulator.CIFr.usrActInd_end then
                    local nll_rewards = self.userScorePred:forward(tabState)
                    lp, rin = torch.max(nll_rewards[opt.lstmHist]:squeeze(), 1)
                    if rin[1] == userRew[opt.lstmHist] then
                        crcRewCnt = crcRewCnt + 1
                        -- update score prediction confusion matrix
                        scorePredTP[rin[1]] = scorePredTP[rin[1]] + 1
                    else
                        scorePredFP[rin[1]] = scorePredFP[rin[1]] + 1
                        scorePredFN[userRew[opt.lstmHist]] = scorePredFN[userRew[opt.lstmHist]] + 1
                    end
                end

                tltCnt = tltCnt + 1

                self.userActsPred:forget()
                self.userScorePred:forget()

            end

--            print('1st act: ', firstActDist)
            print('Following stats are act pred accuracy, reward pred accuracy, act pred accu of each time step')
            print('###', crcActCnt/tltCnt, crcRewCnt/#CIUserActsPred.rnnRealUserDataEnds, torch.cdiv(earlyCrcAct, earlyTotAct))

        else
            -- uSimShLayer == 0 and not lstm models
            for i=1, #CIUserSimulator.realUserDataStates do
                local userState = CIUserSimulator:preprocessUserStateData(CIUserSimulator.realUserDataStates[i], opt.prepro)
                local userAct = CIUserSimulator.realUserDataActs[i]
                local userRew = CIUserSimulator.realUserDataRewards[i]

                local prepUserState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
                prepUserState[1] = userState:clone()

                local nll_acts = self.userActsPred:forward(prepUserState)
                lp, ain = torch.max(nll_acts[1]:squeeze(), 1)

                if i == CIUserSimulator.realUserDataStartLines[userInd]+countScope then firstActDist[ain[1]] = firstActDist[ain[1]]+1 end   -- check act dist at each x-th time step

                -- update action prediction confusion matrix
                if ain[1] == userAct then
                    actPredTP[ain[1]] = actPredTP[ain[1]] + 1
                else
                    actPredFP[ain[1]] = actPredFP[ain[1]] + 1
                    actPredFN[userAct] = actPredFN[userAct] + 1
                end

                --            if ain[1] == userAct[opt.lstmHist] then crcActCnt = crcActCnt + 1 end
                lpy, lps = torch.sort(nll_acts[1]:squeeze(), 1, true)
                local crtExt = false
                local smpLen = self.opt.actEvaScp    -- This is the range of selected actions to check for correct prediction
                for l=1, smpLen do
                    if lps[l] == userAct then
                        crcActCnt = crcActCnt + 1
                        crtExt = true
                        break
                    end
                end

                local indDiff = i - CIUserSimulator.realUserDataStartLines[userInd]
                if indDiff >= 0 and indDiff <= opt.lstmHist+80 and i <= CIUserSimulator.realUserDataEndLines[userInd] then
                    if crtExt then
                        earlyCrcAct[indDiff+1] = earlyCrcAct[indDiff+1] + 1 -- This is at the indDiff's time step, what is prob of correct act prediction
                    end
                    earlyTotAct[indDiff+1] = earlyTotAct[indDiff+1] + 1
                end
                if i == CIUserSimulator.realUserDataEndLines[userInd] then
                    userInd = userInd+1
                end

                if userAct == CIUserSimulator.CIFr.usrActInd_end then
                    local nll_rewards = self.userScorePred:forward(prepUserState)
                    lp, rin = torch.max(nll_rewards[1]:squeeze(), 1)
                    if rin[1] == userRew then
                        crcRewCnt = crcRewCnt + 1
                        -- update score prediction confusion matrix
                        scorePredTP[rin[1]] = scorePredTP[rin[1]] + 1
                    else
                        scorePredFP[rin[1]] = scorePredFP[rin[1]] + 1
                        scorePredFN[userRew] = scorePredFN[userRew] + 1
                    end
                end

                tltCnt = tltCnt + 1

            end

--            print('1st act: ', firstActDist)
            print('Following stats are act pred accuracy, reward pred accuracy, act pred accu of each time step')
            print('###', crcActCnt/tltCnt, crcRewCnt/#CIUserSimulator.realUserDataEndLines, torch.cdiv(earlyCrcAct, earlyTotAct))
        end

    else
        -- opt.uSimShLayer == 1
        -- Unique Act/Score prediction model with shared lower layers
        -- The user action/score predictors for evaluation should be pre-trained, and loaded from files
        -- Also, the CIUserSimulator, CIUserActsPred, CIUserScorePred should be initialized using
        -- the test set.
        print('User Act Score shared-layer model: #', paths.concat(opt.ubgDir , opt.uapFile))
        self.userActScorePred = torch.load(paths.concat(opt.ubgDir , opt.uapFile))
        self.userActScorePred:evaluate()
        self.CIUSim = CIUserSimulator
        self.CIUasp = CIUserActScorePred
        self.opt = opt

        if opt.uppModel == 'lstm' then
            -- uSimShLayer == 1 and lstm model

            --            self._actionDistributionCalc(CIUserSimulator, countScope)

            self.userActScorePred:forget()

            for i=1, #CIUserActScorePred.rnnRealUserDataStates do
                local userState = CIUserActScorePred.rnnRealUserDataStates[i]
                local userAct = CIUserActScorePred.rnnRealUserDataActs[i]
                local userRew = CIUserActScorePred.rnnRealUserDataRewards[i]

                local tabState = {}
                for j=1, opt.lstmHist do
                    local prepUserState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
                    prepUserState[1] = CIUserSimulator:preprocessUserStateData(userState[j], opt.prepro)
                    tabState[j] = prepUserState:clone()
                end

                local nll_acts = self.userActScorePred:forward(tabState)
                lp, ain = torch.max(nll_acts[opt.lstmHist][1]:squeeze(), 1)     -- then 2nd [1] index is for action prediction from the shared act/score prediction outcome
                if i == CIUserActScorePred.rnnRealUserDataStarts[userInd]+countScope then firstActDist[ain[1]] = firstActDist[ain[1]]+1 end   -- check act dist at each x-th time step

                -- update action prediction confusion matrix
                if ain[1] == userAct[opt.lstmHist] then
                    actPredTP[ain[1]] = actPredTP[ain[1]] + 1
                else
                    actPredFP[ain[1]] = actPredFP[ain[1]] + 1
                    actPredFN[userAct[opt.lstmHist]] = actPredFN[userAct[opt.lstmHist]] + 1
                end

                --            if ain[1] == userAct[opt.lstmHist] then crcActCnt = crcActCnt + 1 end
                lpy, lps = torch.sort(nll_acts[opt.lstmHist][1]:squeeze(), 1, true)
                local crtExt = false
                local smpLen = self.opt.actEvaScp    -- This is the range of selected actions to check for correct prediction
                for l=1, smpLen do
                    if lps[l] == userAct[opt.lstmHist] then
                        crcActCnt = crcActCnt + 1
                        crtExt = true
                        break
                    end
                end

                local indDiff = i - CIUserActScorePred.rnnRealUserDataStarts[userInd]
                if indDiff >= 0 and indDiff <= opt.lstmHist+80 and i <= CIUserActScorePred.rnnRealUserDataEnds[userInd] then
                    if crtExt then
                        earlyCrcAct[indDiff+1] = earlyCrcAct[indDiff+1] + 1 -- This is at the indDiff's time step, what is prob of correct act prediction
                    end
                    earlyTotAct[indDiff+1] = earlyTotAct[indDiff+1] + 1
                end
                if i == CIUserActScorePred.rnnRealUserDataEnds[userInd] then
                    userInd = userInd+1
                end

                if userAct[opt.lstmHist] == CIUserSimulator.CIFr.usrActInd_end then
                    -- The predicted reward is the 2nd output of nll_acts in 2nd dim
                    lp, rin = torch.max(nll_acts[opt.lstmHist][2]:squeeze(), 1)
                    if rin[1] == userRew[opt.lstmHist] then
                        crcRewCnt = crcRewCnt + 1
                        -- update score prediction confusion matrix
                        scorePredTP[rin[1]] = scorePredTP[rin[1]] + 1
                    else
                        scorePredFP[rin[1]] = scorePredFP[rin[1]] + 1
                        scorePredFN[userRew[opt.lstmHist]] = scorePredFN[userRew[opt.lstmHist]] + 1
                    end
                end

                tltCnt = tltCnt + 1

                self.userActScorePred:forget()

            end

            --            print('1st act: ', firstActDist)
            print('Following stats are act pred accuracy, reward pred accuracy, act pred accu of each time step')
            print('###', crcActCnt/tltCnt, crcRewCnt/#CIUserActScorePred.rnnRealUserDataEnds, torch.cdiv(earlyCrcAct, earlyTotAct))

        else
            -- SharedLayer == 1, and not lstm models
            for i=1, #CIUserSimulator.realUserDataStates do
                local userState = CIUserSimulator:preprocessUserStateData(CIUserSimulator.realUserDataStates[i], opt.prepro)
                local userAct = CIUserSimulator.realUserDataActs[i]
                local userRew = CIUserSimulator.realUserDataRewards[i]

                local prepUserState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
                prepUserState[1] = userState:clone()

                local nll_acts = self.userActScorePred:forward(prepUserState)

                -- Here, if moe is used with shared lower layers, it is the problem that,
                -- due to limitation of MixtureTable module, we have to join tables together as
                -- a single tensor as output of the whole user action and score prediction model.
                -- So, to guarantee the compatability, we need split the tensor into two tables here,
                -- for act prediction and score prediction respectively.
                if opt.uppModel == 'moe' then
                    nll_acts = nll_acts:split(CIUserSimulator.CIFr.usrActInd_end, 2)  -- We assume 1st dim is batch index. Act pred is the 1st set of output, having dim of 15. Score dim 2.
                end

                lp, ain = torch.max(nll_acts[1][1]:squeeze(), 1)    -- The 1st [1] index means action prediction output from a table, 2nd [1] is batch index, which is not necessary

                if i == CIUserSimulator.realUserDataStartLines[userInd]+countScope then firstActDist[ain[1]] = firstActDist[ain[1]]+1 end   -- check act dist at each x-th time step

                -- update action prediction confusion matrix
                if ain[1] == userAct then
                    actPredTP[ain[1]] = actPredTP[ain[1]] + 1
                else
                    actPredFP[ain[1]] = actPredFP[ain[1]] + 1
                    actPredFN[userAct] = actPredFN[userAct] + 1
                end

                --            if ain[1] == userAct[opt.lstmHist] then crcActCnt = crcActCnt + 1 end
                lpy, lps = torch.sort(nll_acts[1][1]:squeeze(), 1, true)
                local crtExt = false
                local smpLen = self.opt.actEvaScp    -- This is the range of selected actions to check for correct prediction
                for l=1, smpLen do
                    if lps[l] == userAct then
                        crcActCnt = crcActCnt + 1
                        crtExt = true
                        break
                    end
                end

                local indDiff = i - CIUserSimulator.realUserDataStartLines[userInd]
                if indDiff >= 0 and indDiff <= opt.lstmHist+80 and i <= CIUserSimulator.realUserDataEndLines[userInd] then
                    if crtExt then
                        earlyCrcAct[indDiff+1] = earlyCrcAct[indDiff+1] + 1 -- This is at the indDiff's time step, what is prob of correct act prediction
                    end
                    earlyTotAct[indDiff+1] = earlyTotAct[indDiff+1] + 1
                end
                if i == CIUserSimulator.realUserDataEndLines[userInd] then
                    userInd = userInd+1
                end

                if userAct == CIUserSimulator.CIFr.usrActInd_end then
                    -- The predicted reward is the 2nd output of nll_acts in 2nd dim
                    lp, rin = torch.max(nll_acts[2][1]:squeeze(), 1)
                    if rin[1] == userRew then
                        crcRewCnt = crcRewCnt + 1
                        -- update score prediction confusion matrix
                        scorePredTP[rin[1]] = scorePredTP[rin[1]] + 1
                    else
                        scorePredFP[rin[1]] = scorePredFP[rin[1]] + 1
                        scorePredFN[userRew] = scorePredFN[userRew] + 1
                    end
                end

                tltCnt = tltCnt + 1

            end

            --            print('1st act: ', firstActDist)
            print('Following stats are act pred accuracy, reward pred accuracy, act pred accu of each time step')
            print('###', crcActCnt/tltCnt, crcRewCnt/#CIUserSimulator.realUserDataEndLines, torch.cdiv(earlyCrcAct, earlyTotAct))
        end

    end

    -- Calculate Micro and Macro precision, recall and F1 scores
    local actPreMicro = actPredTP:sum() / (actPredTP:sum() + actPredFP:sum())
    local actRecMicro = actPredTP:sum() / (actPredTP:sum() + actPredFN:sum())
    print('Act Prediction Micro Precision: ', actPreMicro, ', Recall: ', actRecMicro, ', F1: ', 2*actPreMicro*actRecMicro/(actPreMicro+actRecMicro))

    local actPreMacro = torch.cdiv(actPredTP, actPredTP + actPredFP):sum() / CIUserSimulator.CIFr.usrActInd_end
    local actRecMacro = torch.cdiv(actPredTP, actPredTP + actPredFN):sum() / CIUserSimulator.CIFr.usrActInd_end
    print('Act Prediction Macro Precision: ', actPreMacro, ', Recall: ', actRecMacro, ', F1: ', 2*actPreMacro*actRecMacro/(actPreMacro+actRecMacro))

    local scorePreMicro = scorePredTP:sum() / (scorePredTP:sum() + scorePredFP:sum())
    local scoreRecMicro = scorePredTP:sum() / (scorePredTP:sum() + scorePredFN:sum())
    print('Score Prediction Micro Precision: ', scorePreMicro, ', Recall: ', scoreRecMicro, ', F1: ', 2*scorePreMicro*scoreRecMicro/(scorePreMicro+scoreRecMicro))

    local scorePreMacro = torch.cdiv(scorePredTP, scorePredTP + scorePredFP):sum() / 2
    local scoreRecMacro = torch.cdiv(scorePredTP, scorePredTP + scorePredFN):sum() / 2
    print('Score Prediction Macro Precision: ', scorePreMacro, ', Recall: ', scoreRecMacro, ', F1: ', 2*scorePreMacro*scoreRecMacro/(scorePreMacro+scoreRecMacro))

end

function CIUserBehaviorGenEvaluator:_actionDistributionCalc(CIUserSimulator, cntScope)
    local countScope = cntScope
    local st = torch.Tensor(CIUserSimulator.CIFr.usrActInd_end):fill(0)     -- tensor dim is 15 (user action types)
    for k, v in ipairs(CIUserSimulator.realUserDataStartLines) do
        st[CIUserSimulator.realUserDataActs[v+countScope]] = st[CIUserSimulator.realUserDataActs[v+countScope]] +1 -- check act dist at each x-th time step
    end
    print('Act count at time step ', countScope+1, ' is:', st)
end

--- Get stats for the over all user action distribution
function CIUserBehaviorGenEvaluator:userActionDistStats(CIUserSimulator)
    local st = torch.Tensor(CIUserSimulator.CIFr.usrActInd_end):fill(0)     -- tensor dim is 15 (user action types)
    for k, v in ipairs(CIUserSimulator.realUserDataActs) do
        st[v] = st[v] +1 -- check act dist at each x-th time step
    end
    local uaSum = st:sum()
    local adist = torch.div(st, uaSum)
    print('User action distribution:\n', 'Total action counts: ', uaSum, '\nAct counts: ', st, '\n dist: ', adist)
end

return CIUserBehaviorGenEvaluator

