--
-- User: pwang8
-- Date: 1/22/17
-- Time: 3:44 PM
-- Using real interaction data to create user simulation model
--

local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'

local CIUserSimulator = classic.class('UserSimulator')

function CIUserSimulator:_init(CIFileReader, opt)
    self.CIFr = CIFileReader    -- a ref to the file reader
    self.realUserDataStates = {}
    self.realUserDataActs = {}
    self.realUserDataRewards = {}
    self.realUserDataStartLines = {}    -- this table stores the starting line of each real human user's interation
    self.realUserDataEndLines = {}

    -- Data structures for the test/train_validation set
    -- This is only used during training, for calculating performance (pred accu) at each epoch
    self.fileReaderTraceDataTest = {}   -- This is the equvalence of CIFileReader, for the test/train_validation set
    self.realUserDataStatesTest = {}
    self.realUserDataActsTest = {}
    self.realUserDataRewardsTest = {}
    self.realUserDataStartLinesTest = {}    -- this table stores the starting line of each real human user's interation
    self.realUserDataEndLinesTest = {}

    -- The following tables are used by rl evaluations
    self.realUserRLStates = {}
    self.realUserRLStatePrepInd = {}
    self.realUserRLActs = {}
    self.realUserRLRewards = {}
    self.realUserRLTerms = {}
    self.realUserRLTypes = {}

    self.userStateFeatureCnt = CIFileReader.userStateGamePlayFeatureCnt + CIFileReader.userStateSurveyFeatureCnt    -- 18+3 now

    self.opt = opt
    -- test for splitting corpus into training and testing
    local ite = 1
    if self.opt.ciuTType == 'train' then
        for userId, userRcd in pairs(CIFileReader.traceData) do
            if ite % 5 == opt.testSetDivSeed then
                self.fileReaderTraceDataTest[userId] = CIFileReader.traceData[userId]   -- save refs in test set
                CIFileReader.traceData[userId] = nil
            end
            ite = ite + 1
        end
    elseif self.opt.ciuTType == 'test' then
        for userId, userRcd in pairs(CIFileReader.traceData) do
            if ite % 5 ~= opt.testSetDivSeed then
                CIFileReader.traceData[userId] = nil
            end
            ite = ite + 1
        end
    elseif self.opt.ciuTType == 'train_tr' then
        for userId, userRcd in pairs(CIFileReader.traceData) do
            if ite % 5 == opt.testSetDivSeed then
                CIFileReader.traceData[userId] = nil
            elseif ite % 5 == opt.validSetDivSeed then
                self.fileReaderTraceDataTest[userId] = CIFileReader.traceData[userId]   -- save refs in train_validation set
                CIFileReader.traceData[userId] = nil
            end
            ite = ite + 1
        end
    elseif self.opt.ciuTType == 'train_ev' then
        for userId, userRcd in pairs(CIFileReader.traceData) do
            if ite % 5 ~= opt.validSetDivSeed then
                CIFileReader.traceData[userId] = nil
            end
            ite = ite + 1
        end
    end

    local above, below = 0, 0
    for userId, userRcd in pairs(CIFileReader.traceData) do
        if CIFileReader.surveyData[userId][CIFileReader.userStateSurveyFeatureCnt+1] > 0.16666667 then
            above = above + 1
        else
            below = below + 1
        end
    end
    print('In '.. self.opt.ciuTType .. ' set, pos pt:', above, ', neg pt:', below)
    -- Use the above method to divide corpus into training and testing, the original corpus is
    -- almost pretty evenly divided wrt nlg/rewards.
    -- The total 402 records corpus has 200 above median nlg records, and 202 equal or below media records.
    -- Using the above splitting method, there are 161 pos, 160 neg in training set, and 39 pos, 42 neg in test set.

    -- The following data processing is on training set
    for userId, userRcd in pairs(CIFileReader.traceData) do

        -- set up initial user state before taking actions
        self.realUserDataStates[#self.realUserDataStates + 1] = torch.Tensor(self.userStateFeatureCnt):fill(0)
        self.realUserDataStartLines[#self.realUserDataStartLines + 1] = #self.realUserDataStates -- Stores start lines for each user interaction
        for i=1, CIFileReader.userStateSurveyFeatureCnt do
            -- set up survey features, which are behind game play features in the state feature tensor
            self.realUserDataStates[#self.realUserDataStates][CIFileReader.userStateGamePlayFeatureCnt+i] = CIFileReader.surveyData[userId][i]
        end

        -- Init RL data table
        self.realUserRLStates[userId] = {}
        self.realUserRLActs[userId] = {}
        self.realUserRLRewards[userId] = {}
        self.realUserRLTerms[userId] = {}
        self.realUserRLTypes[userId] = {}

        for time, act in ipairs(userRcd) do
            self.realUserDataActs[#self.realUserDataStates] = act
--            print('#', userId, self.realUserDataStates[#self.realUserDataStates], ',', self.realUserDataActs[#self.realUserDataStates])

            if CIFileReader.surveyData[userId][CIFileReader.userStateSurveyFeatureCnt+1] > 0.16666667 then  -- above median nlg
                self.realUserDataRewards[#self.realUserDataStates] = 1  -- pos nlg: class_1, neg or 0 nlg: class_2
            else
                self.realUserDataRewards[#self.realUserDataStates] = 2     -- This is (binary) reward class label, not reward value
            end

            if act == CIFileReader.usrActInd_end then
--                print('@@ End action reached')
                if #self.realUserRLTerms[userId] > 0 then
                    -- RL acts in real user data
                    self.realUserRLActs[userId][#self.realUserRLActs[userId]+1] = 1 -- it is fake, just for terminal state. No real action needed
                    -- This is used for constructing realUserRLData tables
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]+1] = self.realUserDataStates[#self.realUserDataStates]:clone()
                    self.realUserRLRewards[userId][#self.realUserRLRewards[userId]+1] = 3 - 2*self.realUserDataRewards[#self.realUserDataStates] -- y = 3-2x
                    self.realUserRLTerms[userId][#self.realUserRLTerms[userId]+1] = 1
                    self.realUserRLTypes[userId][#self.realUserRLTypes[userId]+1] = 0  -- it is a dumb adpType
                else
                    self.realUserRLStates[userId] = nil
                    self.realUserRLActs[userId] = nil
                    self.realUserRLRewards[userId] = nil
                    self.realUserRLTerms[userId] = nil
                    self.realUserRLTypes[userId] = nil
                end
            else
                -- set the next time step state vector
                self.realUserDataStates[#self.realUserDataStates + 1] = self.realUserDataStates[#self.realUserDataStates]:clone()

                if act == CIFileReader.usrActInd_askTeresaSymp then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrStateFeatureInd_TeresaSymp] =
                        (4 - CIFileReader.AdpTeresaSymptomAct[userId][time]) / 3.0  -- (act1--1.0, act3--0.33). So y=(4-x)/3

                    -- RL acts in real user data
                    self.realUserRLActs[userId][#self.realUserRLActs[userId]+1] = CIFileReader.AdpTeresaSymptomAct[userId][time]
                    -- This is used for constructing realUserRLData tables
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]+1] = self.realUserDataStates[#self.realUserDataStates - 1]:clone() -- -1 bcz realUserDataStates has gained a new state, but realUserRLStates needs a old one.
                    self.realUserRLRewards[userId][#self.realUserRLRewards[userId]+1] = 3 - 2*self.realUserDataRewards[#self.realUserDataStates-1] -- y = 3-2x
                    self.realUserRLTerms[userId][#self.realUserRLTerms[userId]+1] = 0
                    -- Add this 1 user's act to RL states, bcz adp act is triggered after this user's act
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]][act] = self.realUserRLStates[userId][#self.realUserRLStates[userId]][act] + 1
                    self.realUserRLTypes[userId][#self.realUserRLTypes[userId]+1] = self.CIFr.ciAdp_TeresaSymp
                elseif act == CIFileReader.usrActInd_askBryceSymp then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrStateFeatureInd_BryceSymp] =
                        (3 - CIFileReader.AdpBryceSymptomAct[userId][time]) / 2.0  -- (act1--1.0, act2--0.5). So y=(3-x)/2

                    -- RL acts in real user data
                    self.realUserRLActs[userId][#self.realUserRLActs[userId]+1] =
                        CIFileReader.AdpBryceSymptomAct[userId][time] + self.CIFr.ciAdpActRange_BryceSymp[1] - 1
                    -- This is used for constructing realUserRLData tables
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]+1] = self.realUserDataStates[#self.realUserDataStates - 1]:clone()
                    self.realUserRLRewards[userId][#self.realUserRLRewards[userId]+1] = 3 - 2*self.realUserDataRewards[#self.realUserDataStates-1] -- y = 3-2x
                    self.realUserRLTerms[userId][#self.realUserRLTerms[userId]+1] = 0
                    -- Add this 1 user's act to RL states, bcz adp act is triggered after this user's act
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]][act] = self.realUserRLStates[userId][#self.realUserRLStates[userId]][act] + 1
                    self.realUserRLTypes[userId][#self.realUserRLTypes[userId]+1] = self.CIFr.ciAdp_BryceSymp
                elseif act == CIFileReader.usrActInd_talkQuentin and
                        self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_KimLetQuentinRevealActOne] < 1 and
                        self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_talkQuentin] < 1 then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrStateFeatureInd_PresentQuiz] =
                            (2 - CIFileReader.AdpPresentQuizAct[userId][time])  -- act1-quiz-1.0, act2-no_quiz-0. y=2-x

                    -- RL acts in real user data
                    self.realUserRLActs[userId][#self.realUserRLActs[userId]+1] =
                        CIFileReader.AdpPresentQuizAct[userId][time] + self.CIFr.ciAdpActRange_PresentQuiz[1] - 1
                    -- This is used for constructing realUserRLData tables
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]+1] = self.realUserDataStates[#self.realUserDataStates - 1]:clone()
                    self.realUserRLRewards[userId][#self.realUserRLRewards[userId]+1] = 3 - 2*self.realUserDataRewards[#self.realUserDataStates-1] -- y = 3-2x
                    self.realUserRLTerms[userId][#self.realUserRLTerms[userId]+1] = 0
                    -- Add this 1 user's act to RL states, bcz adp act is triggered after this user's act
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]][act] = self.realUserRLStates[userId][#self.realUserRLStates[userId]][act] + 1
                    self.realUserRLTypes[userId][#self.realUserRLTypes[userId]+1] = self.CIFr.ciAdp_PresentQuiz
                elseif act == CIFileReader.usrActInd_talkRobert and
                        self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_talkRobert] < 1 then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrStateFeatureInd_PresentQuiz] =
                        (2 - CIFileReader.AdpPresentQuizAct[userId][time])  -- act1-quiz-1.0, act2-no_quiz-0. y=2-x

                    -- RL acts in real user data
                    self.realUserRLActs[userId][#self.realUserRLActs[userId]+1] =
                        CIFileReader.AdpPresentQuizAct[userId][time] + self.CIFr.ciAdpActRange_PresentQuiz[1] - 1
                    -- This is used for constructing realUserRLData tables
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]+1] = self.realUserDataStates[#self.realUserDataStates - 1]:clone()
                    self.realUserRLRewards[userId][#self.realUserRLRewards[userId]+1] = 3 - 2*self.realUserDataRewards[#self.realUserDataStates-1] -- y = 3-2x
                    self.realUserRLTerms[userId][#self.realUserRLTerms[userId]+1] = 0
                    -- Add this 1 user's act to RL states, bcz adp act is triggered after this user's act
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]][act] = self.realUserRLStates[userId][#self.realUserRLStates[userId]][act] + 1
                    self.realUserRLTypes[userId][#self.realUserRLTypes[userId]+1] = self.CIFr.ciAdp_PresentQuiz
                elseif act == CIFileReader.usrActInd_talkFord and
                        self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_talkFord] < 1 then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrStateFeatureInd_PresentQuiz] =
                        (2 - CIFileReader.AdpPresentQuizAct[userId][time])  -- act1-quiz-1.0, act2-no_quiz-0. y=2-x

                    -- RL acts in real user data
                    self.realUserRLActs[userId][#self.realUserRLActs[userId]+1] =
                        CIFileReader.AdpPresentQuizAct[userId][time] + self.CIFr.ciAdpActRange_PresentQuiz[1] - 1
                    -- This is used for constructing realUserRLData tables
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]+1] = self.realUserDataStates[#self.realUserDataStates - 1]:clone()
                    self.realUserRLRewards[userId][#self.realUserRLRewards[userId]+1] = 3 - 2*self.realUserDataRewards[#self.realUserDataStates-1] -- y = 3-2x
                    self.realUserRLTerms[userId][#self.realUserRLTerms[userId]+1] = 0
                    -- Add this 1 user's act to RL states, bcz adp act is triggered after this user's act
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]][act] = self.realUserRLStates[userId][#self.realUserRLStates[userId]][act] + 1
                    self.realUserRLTypes[userId][#self.realUserRLTypes[userId]+1] = self.CIFr.ciAdp_PresentQuiz
                elseif act == CIFileReader.usrActInd_submitWorksheet then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrStateFeatureInd_WorksheetLevel] =
                        (CIFileReader.AdpWorksheetLevelAct[userId][time] / 3.0)  -- act1-0.33, act3-1. y=x/3

                    -- RL acts in real user data
                    self.realUserRLActs[userId][#self.realUserRLActs[userId]+1] =
                        CIFileReader.AdpWorksheetLevelAct[userId][time] + self.CIFr.ciAdpActRange_WorksheetLevel[1] - 1
                    -- This is used for constructing realUserRLData tables
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]+1] = self.realUserDataStates[#self.realUserDataStates - 1]:clone()
                    self.realUserRLRewards[userId][#self.realUserRLRewards[userId]+1] = 3 - 2*self.realUserDataRewards[#self.realUserDataStates-1] -- y = 3-2x
                    self.realUserRLTerms[userId][#self.realUserRLTerms[userId]+1] = 0
                    -- Add this 1 user's act to RL states, bcz adp act is triggered after this user's act
                    self.realUserRLStates[userId][#self.realUserRLStates[userId]][act] = self.realUserRLStates[userId][#self.realUserRLStates[userId]][act] + 1
                    self.realUserRLTypes[userId][#self.realUserRLTypes[userId]+1] = self.CIFr.ciAdp_WorksheetLevel
                end

                -- Add 1 to corresponding state features
                self.realUserDataStates[#self.realUserDataStates][act] = self.realUserDataStates[#self.realUserDataStates][act] + 1

                -- For indices 12, 13, 14, state feature values can only be 0 or 1
                if self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_BryceRevealActOne] > 1 then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_BryceRevealActOne] = 1
                end
                if self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_QuentinRevealActOne] > 1 then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_QuentinRevealActOne] = 1
                end
                if self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_KimLetQuentinRevealActOne] > 1 then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_KimLetQuentinRevealActOne] = 1
                end

            end

        end

    end
    print('Human user actions number: ', #self.realUserDataStates, #self.realUserDataActs)

    for i=1, #self.realUserDataStartLines - 1 do
        self.realUserDataEndLines[i] = self.realUserDataStartLines[i+1] - 1
    end
    self.realUserDataEndLines[#self.realUserDataStartLines] = #self.realUserDataStates


    if self.opt.ciuTType == 'train' or self.opt.ciuTType == 'train_tr' then
        -- The following data processing is on test/train_validation set
        for userId, userRcd in pairs(self.fileReaderTraceDataTest) do

            -- set up initial user state before taking actions
            self.realUserDataStatesTest[#self.realUserDataStatesTest + 1] = torch.Tensor(self.userStateFeatureCnt):fill(0)
            self.realUserDataStartLinesTest[#self.realUserDataStartLinesTest + 1] = #self.realUserDataStatesTest -- Stores start lines for each user interaction (in test set)
            for i=1, CIFileReader.userStateSurveyFeatureCnt do
                -- set up survey features, which are behind game play features in the state feature tensor
                self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.userStateGamePlayFeatureCnt+i] = CIFileReader.surveyData[userId][i]
            end

            for time, act in ipairs(userRcd) do
                self.realUserDataActsTest[#self.realUserDataStatesTest] = act

                if CIFileReader.surveyData[userId][CIFileReader.userStateSurveyFeatureCnt+1] > 0.16666667 then  -- above median nlg
                    self.realUserDataRewardsTest[#self.realUserDataStatesTest] = 1  -- pos nlg: class_1, neg or 0 nlg: class_2
                else
                    self.realUserDataRewardsTest[#self.realUserDataStatesTest] = 2     -- This is (binary) reward class label, not reward value
                end

                if act ~= CIFileReader.usrActInd_end then
                    -- set the next time step state set
                    self.realUserDataStatesTest[#self.realUserDataStatesTest + 1] = self.realUserDataStatesTest[#self.realUserDataStatesTest]:clone()

                    if act == CIFileReader.usrActInd_askTeresaSymp then
                        self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrStateFeatureInd_TeresaSymp] =
                        (4 - CIFileReader.AdpTeresaSymptomAct[userId][time]) / 3.0  -- (act1--1.0, act3--0.33). So y=(4-x)/3

                    elseif act == CIFileReader.usrActInd_askBryceSymp then
                        self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrStateFeatureInd_BryceSymp] =
                        (3 - CIFileReader.AdpBryceSymptomAct[userId][time]) / 2.0  -- (act1--1.0, act2--0.5). So y=(3-x)/2

                    elseif act == CIFileReader.usrActInd_talkQuentin and
                            self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrActInd_KimLetQuentinRevealActOne] < 1 and
                            self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrActInd_talkQuentin] < 1 then
                        self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrStateFeatureInd_PresentQuiz] =
                        (2 - CIFileReader.AdpPresentQuizAct[userId][time])  -- act1-quiz-1.0, act2-no_quiz-0. y=2-x

                    elseif act == CIFileReader.usrActInd_talkRobert and
                            self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrActInd_talkRobert] < 1 then
                        self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrStateFeatureInd_PresentQuiz] =
                        (2 - CIFileReader.AdpPresentQuizAct[userId][time])  -- act1-quiz-1.0, act2-no_quiz-0. y=2-x

                    elseif act == CIFileReader.usrActInd_talkFord and
                            self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrActInd_talkFord] < 1 then
                        self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrStateFeatureInd_PresentQuiz] =
                        (2 - CIFileReader.AdpPresentQuizAct[userId][time])  -- act1-quiz-1.0, act2-no_quiz-0. y=2-x

                    elseif act == CIFileReader.usrActInd_submitWorksheet then
                        self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrStateFeatureInd_WorksheetLevel] =
                        (CIFileReader.AdpWorksheetLevelAct[userId][time] / 3.0)  -- act1-0.33, act3-1. y=x/3

                    end

                    -- Add 1 to corresponding state features
                    self.realUserDataStatesTest[#self.realUserDataStatesTest][act] = self.realUserDataStatesTest[#self.realUserDataStatesTest][act] + 1

                    -- For indices 12, 13, 14, state feature values can only be 0 or 1
                    if self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrActInd_BryceRevealActOne] > 1 then
                        self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrActInd_BryceRevealActOne] = 1
                    end
                    if self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrActInd_QuentinRevealActOne] > 1 then
                        self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrActInd_QuentinRevealActOne] = 1
                    end
                    if self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrActInd_KimLetQuentinRevealActOne] > 1 then
                        self.realUserDataStatesTest[#self.realUserDataStatesTest][CIFileReader.usrActInd_KimLetQuentinRevealActOne] = 1
                    end

                end

            end

        end
        print('Human user actions number in test set: ', #self.realUserDataStatesTest, #self.realUserDataActsTest)

        for i=1, #self.realUserDataStartLinesTest - 1 do
            self.realUserDataEndLinesTest[i] = self.realUserDataStartLinesTest[i+1] - 1
        end
        self.realUserDataEndLinesTest[#self.realUserDataStartLinesTest] = #self.realUserDataStatesTest
    end


    self.stateFeatureRescaleFactor = torch.Tensor(self.userStateFeatureCnt):fill(1)
    self.stateFeatureMeanEachFeature = torch.Tensor(self.userStateFeatureCnt):fill(0)
    self.stateFeatureStdEachFeature = torch.Tensor(self.userStateFeatureCnt):fill(1)
    -- Calculate user state feature value rescale factors
    self:_calcRealUserStateFeatureRescaleFactor()


    -- These RL format data are used in RL agent training directly using raw data
    for uid, uRLStates in pairs(self.realUserRLStates) do
        self.realUserRLStatePrepInd[uid] = {}
        for k, v in pairs(uRLStates) do
            local ruRLStatePrep = torch.Tensor(1, 1, self.userStateFeatureCnt):fill(0)   -- this state should be 3d
            ruRLStatePrep[1][1] = self:preprocessUserStateData(v, self.opt.prepro)   -- do preprocessing before sending back to RL
            self.realUserRLStatePrepInd[uid][k] = torch.Tensor(1, 1, self.userStateFeatureCnt + #self.CIFr.ciAdpActRanges):fill(0)

            if self.realUserRLTerms[uid][k] == 1 then     -- if terminal
                self:_updateRLStatePrepTypeInd(self.realUserRLStatePrepInd[uid][k],
                    ruRLStatePrep, self.realUserRLTypes[uid][k], true)
            else
                self:_updateRLStatePrepTypeInd(self.realUserRLStatePrepInd[uid][k],
                    ruRLStatePrep, self.realUserRLTypes[uid][k], false)
            end
        end
    end

    collectgarbage()

    -- The shortest length record has a user actoin sequence length of 2. User id is 100-0466
--    -- calc min length
--    local minlen = 9999
--    for i=1,#self.realUserDataStartLines-1 do
--        if minlen > self.realUserDataStartLines[i+1] - self.realUserDataStartLines[i] then
--            minlen = self.realUserDataStartLines[i+1] - self.realUserDataStartLines[i]
--        end
--    end
--    if minlen > #self.realUserDataStates - self.realUserDataStartLines[#self.realUserDataStartLines] then
--        minlen = #self.realUserDataStates - self.realUserDataStartLines[#self.realUserDataStartLines]
--    end
--    print('$$$$$ min traj length is', minlen) os.exit()
    -- 273 students with postive nlg, 39 with 0 nlg, 90 with negative nlg. 67.9%

    -- The following code show the real user data in rl format, which can be used in evaluation directly
--    print('Testing real user rl data', TableSet.countsInSet(self.realUserRLStates),
--        TableSet.countsInSet(self.realUserRLActs),
--        TableSet.countsInSet(self.realUserRLRewards),
--        TableSet.countsInSet(self.realUserRLTerms),
--        TableSet.countsInSet(self.realUserRLTypes),
--        TableSet.countsInSet(self.realUserRLStatePrepInd))
--
--
--    local i=1
--    for k,v in pairs(self.realUserRLStates) do
--        print('k',k, self.realUserRLActs[k], self.realUserRLRewards[k], self.realUserRLTerms[k], self.realUserRLTypes[k])
--        for time, sta in pairs(v) do
--            print('state:', sta, '\n prep state:', self.realUserRLStatePrepInd[k][time])
--        end
--        print('#####')
--        i = i+1
--        if i==3 then break end
--    end

    --- The following tensors are used to calculated pearson's correlations between state features
    self.featSqre = torch.Tensor(self.userStateFeatureCnt):fill(0)
    self.featCrossSqre = torch.Tensor(self.userStateFeatureCnt, self.userStateFeatureCnt):fill(0)

end


function CIUserSimulator:_updateRLStatePrepTypeInd(rlStatePrepTypeInd, rlStatePrep, adpType, endingType)
    rlStatePrepTypeInd:zero()
    for i=1, rlStatePrep:size(3) do
        rlStatePrepTypeInd[1][1][i] = rlStatePrep[1][1][i]
    end
    if not endingType then
        rlStatePrepTypeInd[1][1][rlStatePrep:size(3) + adpType] = 1
    end
end

--- Calculate the observed largest state feature value for each game play feature,
--- and use it to rescale feature value later
function CIUserSimulator:_calcRealUserStateFeatureRescaleFactor()
    local allUserDataStates = torch.Tensor(#self.realUserDataStates, self.userStateFeatureCnt)
    local allInd = 1
    for _,v in pairs(self.realUserDataStates) do
        for i=1, self.CIFr.userStateGamePlayFeatureCnt do
            if self.stateFeatureRescaleFactor[i] < v[i] then
                self.stateFeatureRescaleFactor[i] = v[i]
            end
        end
        allUserDataStates[allInd] = v:clone()
        allInd = allInd + 1
    end
    self.stateFeatureMeanEachFeature = torch.mean(allUserDataStates, 1):squeeze()
    self.stateFeatureStdEachFeature = torch.std(allUserDataStates, 1):squeeze()
--    print('@@', self.stateFeatureMeanEachFeature, '#', self.stateFeatureStdEachFeature)
--    print('##', self.stateFeatureRescaleFactor)
    -- For the 402 CI data, this stateFeatureRescaleFactor vector is
    -- {44 ,20 ,3 ,9 ,7 ,9 ,7 ,10 ,39 ,43 ,10 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1}
    -- Note: when real/simulated user data is used in ML algorithms,
    -- the raw feature values should be divided by this tensor for rescaling.
    -- torch.cdiv(x, self.stateFeatureRescaleFactor)
    -- This is the stateFeatureMeanEachFeature:
    -- {5.0325, 1.3671, 0.7945, 0.9831, 1.0969, 1.6381, 0.9424, 0.8351, 2.9051,
    -- 5.3980, 0.6525, 0.2551, 0.2581, 0.1637, 0.4778, 0.5266, 0.2398, 0.4211, 0.4642 ,0.6148 ,0.3487}
    -- This is the stateFeatureStdEachFeature:
    -- {4.7120, 2.4729, 0.5952, 0.9774, 1.0633, 1.4782, 0.8478, 0.8897, 5.1906, 6.7046,
    -- 1.1095, 0.4359, 0.4376, 0.3700, 0.3904, 0.4136, 0.3589, 0.4938, 0.4987, 0.2325, 0.1198}
end

--- Right now, this preprocessing is rescaling
function CIUserSimulator:preprocessUserStateData(obvUserData, ppType)

    -- Attention: We are right now doing a training/test split. So, all preprocessUserStateData
    -- should use training set data. So, just directly put it here for convenience. Not a good sln.
    self.stateFeatureMeanEachFeature = torch.Tensor{
        5.2252, 1.3964, 0.7949, 1.0003, 1.1239, 1.5931, 0.9404,
        0.8490, 2.8132, 5.4015, 0.6394, 0.2554, 0.2510, 0.1572,
        0.4907, 0.5385, 0.2344, 0.4313, 0.4647, 0.6142, 0.3467
    }

    self.stateFeatureStdEachFeature = torch.Tensor{
        4.8425, 2.5111, 0.5854, 0.9263, 1.0933, 1.4194, 0.8491,
        0.8952, 5.0249, 6.6338, 1.1148, 0.4361, 0.4336, 0.3640,
        0.3896, 0.4155, 0.3564, 0.4953, 0.4988, 0.2322, 0.1177
    }

    if ppType == 'rsc' then
        return torch.cdiv(obvUserData, self.stateFeatureRescaleFactor)
    elseif ppType == 'std' then
        local subMean = torch.add(obvUserData, -1, self.stateFeatureMeanEachFeature)
        return torch.cdiv(subMean, self.stateFeatureStdEachFeature)
    else
        print('!!!Error. Unrecognized preprocessing in UserSimulator.', ppType)
    end

end

--- Check if narrative adaptation point will be triggered
--  Notice: the curState should be raw state values, not preprocessed values
--  Attention: This act param is user's action, not RL agent's adp action
function CIUserSimulator:isAdpTriggered(curState, userAct)
    -- curState should be a 1d or 2d tensor. If it is
    -- 2d, I assume the 1st dim is batch dimension
    assert(curState:dim() == 1 or curState:dim() == 2)
    local stateRef = curState
    if curState:dim() == 2 then stateRef = curState[1] end

    if userAct == self.CIFr.usrActInd_askTeresaSymp then
        return true, self.CIFr.ciAdp_TeresaSymp
    elseif userAct == self.CIFr.usrActInd_askBryceSymp then
        return true, self.CIFr.ciAdp_BryceSymp
    elseif userAct == self.CIFr.usrActInd_talkQuentin and
            stateRef[self.CIFr.usrActInd_KimLetQuentinRevealActOne] < 1 and
            stateRef[self.CIFr.usrActInd_talkQuentin] < 1 then
        return true, self.CIFr.ciAdp_PresentQuiz
    elseif userAct == self.CIFr.usrActInd_talkRobert and
            stateRef[self.CIFr.usrActInd_talkRobert] < 1 then
        return true, self.CIFr.ciAdp_PresentQuiz
    elseif userAct == self.CIFr.usrActInd_talkFord and
            stateRef[self.CIFr.usrActInd_talkFord] < 1 then
        return true, self.CIFr.ciAdp_PresentQuiz
    elseif userAct == self.CIFr.usrActInd_submitWorksheet then
        return true, self.CIFr.ciAdp_WorksheetLevel
    end

    return false, 0
end


--- Apply user action on state representation
--- Attention: isAdpTriggered() should be called
--- before this function. (Verification of adaptation should be before applying user action's effect)
--- Attention: This act param is user's action, not RL agent's adp action
function CIUserSimulator:applyUserActOnState(curState, userAct)
    -- curState should be a 1d or 2d tensor. If it is
    -- 2d, I assume the 1st dim is batch dimension
    assert(userAct >= self.CIFr.usrActInd_posterRead and userAct < self.CIFr.usrActInd_end)
    assert(curState:dim() == 1 or curState:dim() == 2 or curState:dim() == 3)
    local stateRef = curState
    if curState:dim() == 2 then
        stateRef = curState[1]
    elseif curState:dim() == 3 then
        stateRef = curState[1][1]
    end

    -- Add 1 to corresponding state features
    stateRef[userAct] = stateRef[userAct] + 1

    -- For indices 12, 13, 14, state feature values can only be 0 or 1
    if stateRef[self.CIFr.usrActInd_BryceRevealActOne] > 1 then
        stateRef[self.CIFr.usrActInd_BryceRevealActOne] = 1
    end
    if stateRef[self.CIFr.usrActInd_QuentinRevealActOne] > 1 then
        stateRef[self.CIFr.usrActInd_QuentinRevealActOne] = 1
    end
    if stateRef[self.CIFr.usrActInd_KimLetQuentinRevealActOne] > 1 then
        stateRef[self.CIFr.usrActInd_KimLetQuentinRevealActOne] = 1
    end

    return curState
end


--- Apply RL agent Adaptation action on state representation
--  Attention: This act param is RL agent's action, not user's action
function CIUserSimulator:applyAdpActOnState(curState, adpType, adpAct)
    -- curState should be a 1d or 2d tensor. If it is
    -- 2d, I assume the 1st dim is batch dimension
    assert(curState:dim() == 1 or curState:dim() == 2)
    local stateRef = curState
    if curState:dim() == 2 then
        stateRef = curState[1]
    end

    if adpType == self.CIFr.ciAdp_TeresaSymp then
        stateRef[self.CIFr.usrStateFeatureInd_TeresaSymp] =
        (4 - adpAct) / 3.0  -- (act1--1.0, act3--0.33). So y=(4-x)/3. Act: 1-3
    elseif adpType == self.CIFr.ciAdp_BryceSymp then
        stateRef[self.CIFr.usrStateFeatureInd_BryceSymp] =
        (3 - (adpAct - self.CIFr.ciAdpActRange_BryceSymp[1]+1)) / 2.0  -- (act1--1.0, act2--0.5). So y=(3-x)/2. Act: 4-5
    elseif adpType == self.CIFr.ciAdp_PresentQuiz then
        stateRef[self.CIFr.usrStateFeatureInd_PresentQuiz] =
        (2 - (adpAct - self.CIFr.ciAdpActRange_PresentQuiz[1]+1))  -- act1-quiz-1.0, act2-no_quiz-0. y=2-x. Act: 9-10
    elseif adpType == self.CIFr.ciAdp_WorksheetLevel then
        stateRef[self.CIFr.usrStateFeatureInd_WorksheetLevel] =
        ((adpAct - self.CIFr.ciAdpActRange_WorksheetLevel[1]+1) / 3.0)  -- act1-0.33, act3-1. y=x/3. Act: 6-8
    end

    return curState
end

-- Description: Pearson correlation coefficient
function CIUserSimulator:PearsonCorr()
    ---- compute the mean
    --local x1, y1 = 0, 0
    --for _, v in pairs(a) do
    --    x1, y1 = x1 + v[1], y1 + v[2]
    --end
    ---- compute the coefficient
    --x1, y1 = x1 / #a, y1 / #a
    --local x2, y2, xy = 0, 0, 0
    --for _, v in pairs(a) do
    --    local tx, ty = v[1] - x1, v[2] - y1
    --    xy, x2, y2 = xy + tx * ty, x2 + tx * tx, y2 + ty * ty
    --end
    --return xy / math.sqrt(x2) / math.sqrt(y2)

    -- The input should be ciUserSimulator.realUserDataStates
    -- So, here we have all input feature values in all data points
    local stateDim = self.realUserDataStates[1]:size()[1]
    local stateMean = torch.Tensor(stateDim):zero()
    for i=1, #self.realUserDataStates do
        stateMean = stateMean + self.realUserDataStates[i]
    end
    stateMean:div(#self.realUserDataStates)

    local diffWithMean = torch.Tensor(#self.realUserDataStates, stateDim):zero()
    for i=1, #self.realUserDataStates do
        diffWithMean[i] = self.realUserDataStates[i] - stateMean
    end

    local allEleSqr
    for j=1, stateDim do
        -- get squared values of all features in all standardized data points
        allEleSqr = torch.pow(diffWithMean, 2)
    end

    -- Get sum over all data points, for each feature
    self.featSqre = torch.sum(allEleSqr, 1)

    -- Time to calculate the a*b. Consider to use tensor:sub, or narrow, or select functions.
    --for j=1, stateDim do
    --    for k=j, stateDim do
    --
    --    end
    --end

end

return CIUserSimulator
