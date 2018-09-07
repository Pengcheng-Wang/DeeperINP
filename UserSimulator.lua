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
    self.realUserDataStandardNLG = {}   -- std for nlg is 0.27016 for 402 records
    self.realUserDataStartLines = {}    -- this table stores the starting line of each real human user's interation
    self.realUserDataEndLines = {}

    -- Data structures for the test/train_validation set
    -- This is only used during training, for calculating performance (pred accu) at each epoch
    self.fileReaderTraceDataTest = {}   -- This is the equvalence of CIFileReader, for the test/train_validation set
    self.realUserDataStatesTest = {}
    self.realUserDataActsTest = {}
    self.realUserDataRewardsTest = {}
    self.realUserDataStandardNLGTest = {}
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

    if self.opt.trainTwoFoldSim < 0.5 then
        -- Normally prepare data for 5-fold cross-validation
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
    else
        -- Prepare data for 2-fold cross-validation. This can be used in DRL model evaluation
        assert(self.opt.ciuTType == 'train', 'Only supports train mode when prepare player simulator for 2-fold cross-validation')
        assert(opt.testSetDivSeed == 0 or opt.testSetDivSeed == 1, 'Now we simply do 2-fold cross-validation')
        print('Train player simulator for 2-fold cross-validation.')
        for userId, userRcd in pairs(CIFileReader.traceData) do
            if ite % 2 == opt.testSetDivSeed then
                self.fileReaderTraceDataTest[userId] = CIFileReader.traceData[userId]   -- save refs in test set
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
            self.realUserDataStandardNLG[#self.realUserDataStates] =
            (CIFileReader.surveyData[userId][CIFileReader.userStateSurveyFeatureCnt+1] - 0.16666667) / 0.27016  -- 1.5 Dec 17,2017. I changed this part. Because it is not the correct way to weight errors from score regression source.    --0.27016   -- 0.27016 is the std of nlg

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
                self.realUserDataStandardNLGTest[#self.realUserDataStatesTest] =
                (CIFileReader.surveyData[userId][CIFileReader.userStateSurveyFeatureCnt+1] - 0.16666667) / 0.27016  -- 1.5 Dec 17,2017. I changed this part. Because it is not the correct way to weight errors from score regression source.    --0.27016   -- 0.27016 is the std of nlg

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

    -- The shortest length record has a user action sequence length of 2. User id is 100-0466
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


    ----------------------------------------------------------------------
    -- modified by @pwang8, on Sept 6, 2018.
    -- The following code snippet is used to save
    --  self.realUserDataStates
    --  self.realUserDataActs
    --  self.realUserDataRewards
    -- &
    --  self.realUserDataStatesTest
    --  self.realUserDataActsTest
    --  self.realUserDataRewardsTest
    -- into a csv file. These information will be used to construct act/score
    -- predictors using other libs like sklearn
    local usrFile = io.open('./userModelTrained/userStateActOnly.csv', 'w')
    usrFile:write('A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,Act\n') -- Column name, A-U are 21 features in state representation
    for i=1, #self.realUserDataStates do
        local i_klen = self.realUserDataStates[i]:size(1)
        for i_k=1, i_klen do
            usrFile:write(self.realUserDataStates[i][i_k], ',')
        end
        usrFile:write(self.realUserDataActs[i]-1, '\n')
    end
    for i=1, #self.realUserDataStatesTest do
        local i_klen = self.realUserDataStatesTest[i]:size(1)
        for i_k=1, i_klen do
            usrFile:write(self.realUserDataStatesTest[i][i_k], ',')
        end
        usrFile:write(self.realUserDataActsTest[i]-1, '\n')
    end
    usrFile:close()

    -- Following code writes reward (final time reward) into an csv file
    usrFile = io.open('./userModelTrained/userStateRewardOnly.csv', 'w')
    usrFile:write('A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,Reward\n') -- Column name, A-U are 21 features in state representation
    for i=1, #self.realUserDataEndLines do
        local i_klen = self.realUserDataStates[1]:size(1)
        for i_k=1, i_klen do
            usrFile:write(self.realUserDataStates[self.realUserDataEndLines[i]][i_k], ',')
        end
        usrFile:write(self.realUserDataRewards[self.realUserDataEndLines[i]]-1, '\n')
    end
    for i=1, #self.realUserDataEndLinesTest do
        local i_klen = self.realUserDataStatesTest[1]:size(1)
        for i_k=1, i_klen do
            usrFile:write(self.realUserDataStatesTest[self.realUserDataEndLinesTest[i]][i_k], ',')
        end
        usrFile:write(self.realUserDataRewardsTest[self.realUserDataEndLinesTest[i]]-1, '\n')
    end
    usrFile:close()
    os.exit()
    -- This is just temporal



    ----------------------------------------------------------------------
    --- Prepare data for RNN models in training set
    ---
    self.rnnRealUserDataStates = {}
    self.rnnRealUserDataActs = {}
    self.rnnRealUserDataRewards = {}
    self.rnnRealUserDataStandardNLG = {}
    self.rnnRealUserDataStarts = {}
    self.rnnRealUserDataEnds = {}
    self.rnnRealUserDataPad = torch.Tensor(#self.realUserDataStartLines):fill(0)    -- indicating whether data has padding at head (should be padded)
    if string.sub(opt.uppModel, 1, 4) == 'rnn_' then
        local indSeqHead = 1
        local indSeqTail = opt.lstmHist
        local indUserSeq = 1    -- user id ptr. Use this to get the tail of each trajectory
        while indSeqTail <= #self.realUserDataStates do
            if self.rnnRealUserDataPad[indUserSeq] < 1 then
                for padi = opt.lstmHist-1, 1, -1 do
                    self.rnnRealUserDataStates[#self.rnnRealUserDataStates + 1] = {}
                    self.rnnRealUserDataActs[#self.rnnRealUserDataActs + 1] = {}
                    self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards + 1] = {}
                    self.rnnRealUserDataStandardNLG[#self.rnnRealUserDataStandardNLG + 1] = {}
                    for i=1, padi do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i] = torch.Tensor(self.userStateFeatureCnt):fill(0)
                        self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i] = self.realUserDataActs[indSeqHead]  -- duplicate the 1st user action for padded states
                        self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards][i] = self.realUserDataRewards[indSeqHead]
                        self.rnnRealUserDataStandardNLG[#self.rnnRealUserDataStandardNLG][i] = self.realUserDataStandardNLG[indSeqHead]
                    end
                    for i=1, opt.lstmHist-padi do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i+padi] = self.realUserDataStates[indSeqHead+i-1]
                        self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i+padi] = self.realUserDataActs[indSeqHead+i-1]
                        self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards][i+padi] = self.realUserDataRewards[indSeqHead+i-1]
                        self.rnnRealUserDataStandardNLG[#self.rnnRealUserDataStandardNLG][i+padi] = self.realUserDataStandardNLG[indSeqHead+i-1]
                    end
                    if padi == opt.lstmHist-1 then
                        self.rnnRealUserDataStarts[#self.rnnRealUserDataStarts+1] = #self.rnnRealUserDataStates     -- This is the start of a user's record -- This is duplicated. The value should be the same as realUserDataStartLines
                    end
                    if indSeqHead+(opt.lstmHist-padi)-1 == self.realUserDataEndLines[indUserSeq] then
                        self.rnnRealUserDataPad[indUserSeq] = 1
                        break   -- if padding tail is going to outrange this user record's tail, break
                    end
                end
                self.rnnRealUserDataPad[indUserSeq] = 1
            else
                if indSeqTail <= self.realUserDataEndLines[indUserSeq] then
                    self.rnnRealUserDataStates[#self.rnnRealUserDataStates + 1] = {}
                    self.rnnRealUserDataActs[#self.rnnRealUserDataActs + 1] = {}
                    self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards + 1] = {}
                    self.rnnRealUserDataStandardNLG[#self.rnnRealUserDataStandardNLG + 1] = {}
                    for i=1, opt.lstmHist do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i] = self.realUserDataStates[indSeqHead+i-1]
                        self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i] = self.realUserDataActs[indSeqHead+i-1]
                        self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards][i] = self.realUserDataRewards[indSeqHead+i-1]
                        self.rnnRealUserDataStandardNLG[#self.rnnRealUserDataStandardNLG][i] = self.realUserDataStandardNLG[indSeqHead+i-1]
                    end
                    indSeqHead = indSeqHead + 1
                    indSeqTail = indSeqTail + 1
                else
                    self.rnnRealUserDataEnds[#self.rnnRealUserDataEnds+1] = #self.rnnRealUserDataStates     -- This is the end of a user's record
                    indUserSeq = indUserSeq + 1 -- next user's records
                    indSeqHead = self.realUserDataStartLines[indUserSeq]
                    indSeqTail = indSeqHead + opt.lstmHist - 1
                end
            end
        end
        self.rnnRealUserDataEnds[#self.rnnRealUserDataEnds+1] = #self.rnnRealUserDataStates     -- Set the end of the last user's record
        -- There are in total 15509 sequences if histLen is 3. 14707 if histLen is 5. 15108 if histLen is 4. 15911 if histLen is 2.
    end

    ----------------------------------------------------------------------
    --- Prepare data for RNN models in test/train_validation set
    ---
    self.rnnRealUserDataStatesTest = {}
    self.rnnRealUserDataActsTest = {}
    self.rnnRealUserDataRewardsTest = {}
    self.rnnRealUserDataStandardNLGTest = {}
    self.rnnRealUserDataStartsTest = {}
    self.rnnRealUserDataEndsTest = {}
    self.rnnRealUserDataPadTest = torch.Tensor(#self.realUserDataStartLinesTest):fill(0)    -- indicating whether data has padding at head (should be padded)
    if self.opt.ciuTType == 'train' or self.opt.ciuTType == 'train_tr' then
        if string.sub(opt.uppModel, 1, 4) == 'rnn_' then
            local indSeqHead = 1
            local indSeqTail = opt.lstmHist
            local indUserSeq = 1    -- user id ptr. Use this to get the tail of each trajectory
            while indSeqTail <= #self.realUserDataStatesTest do
                if self.rnnRealUserDataPadTest[indUserSeq] < 1 then
                    for padi = opt.lstmHist-1, 1, -1 do
                        self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest + 1] = {}
                        self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest + 1] = {}
                        self.rnnRealUserDataRewardsTest[#self.rnnRealUserDataRewardsTest + 1] = {}
                        self.rnnRealUserDataStandardNLGTest[#self.rnnRealUserDataStandardNLGTest + 1] = {}
                        for i=1, padi do
                            self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest][i] = torch.Tensor(self.userStateFeatureCnt):fill(0)
                            self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest][i] = self.realUserDataActsTest[indSeqHead]  -- duplicate the 1st user action for padded states
                            self.rnnRealUserDataRewardsTest[#self.rnnRealUserDataRewardsTest][i] = self.realUserDataRewardsTest[indSeqHead]
                            self.rnnRealUserDataStandardNLGTest[#self.rnnRealUserDataStandardNLGTest][i] = self.realUserDataStandardNLGTest[indSeqHead]
                        end
                        for i=1, opt.lstmHist-padi do
                            self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest][i+padi] = self.realUserDataStatesTest[indSeqHead+i-1]
                            self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest][i+padi] = self.realUserDataActsTest[indSeqHead+i-1]
                            self.rnnRealUserDataRewardsTest[#self.rnnRealUserDataRewardsTest][i+padi] = self.realUserDataRewardsTest[indSeqHead+i-1]
                            self.rnnRealUserDataStandardNLGTest[#self.rnnRealUserDataStandardNLGTest][i+padi] = self.realUserDataStandardNLGTest[indSeqHead+i-1]
                        end
                        if padi == opt.lstmHist-1 then
                            self.rnnRealUserDataStartsTest[#self.rnnRealUserDataStartsTest+1] = #self.rnnRealUserDataStatesTest     -- This is the start of a user's record
                        end
                        if indSeqHead+(opt.lstmHist-padi)-1 == self.realUserDataEndLinesTest[indUserSeq] then
                            self.rnnRealUserDataPadTest[indUserSeq] = 1
                            break   -- if padding tail is going to outrange this user record's tail, break
                        end
                    end
                    self.rnnRealUserDataPadTest[indUserSeq] = 1
                else
                    if indSeqTail <= self.realUserDataEndLinesTest[indUserSeq] then
                        self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest + 1] = {}
                        self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest + 1] = {}
                        self.rnnRealUserDataRewardsTest[#self.rnnRealUserDataRewardsTest + 1] = {}
                        self.rnnRealUserDataStandardNLGTest[#self.rnnRealUserDataStandardNLGTest + 1] = {}
                        for i=1, opt.lstmHist do
                            self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest][i] = self.realUserDataStatesTest[indSeqHead+i-1]
                            self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest][i] = self.realUserDataActsTest[indSeqHead+i-1]
                            self.rnnRealUserDataRewardsTest[#self.rnnRealUserDataRewardsTest][i] = self.realUserDataRewardsTest[indSeqHead+i-1]
                            self.rnnRealUserDataStandardNLGTest[#self.rnnRealUserDataStandardNLGTest][i] = self.realUserDataStandardNLGTest[indSeqHead+i-1]
                        end
                        indSeqHead = indSeqHead + 1
                        indSeqTail = indSeqTail + 1
                    else
                        self.rnnRealUserDataEndsTest[#self.rnnRealUserDataEndsTest+1] = #self.rnnRealUserDataStatesTest     -- This is the end of a user's record
                        indUserSeq = indUserSeq + 1 -- next user's records
                        indSeqHead = self.realUserDataStartLinesTest[indUserSeq]
                        indSeqTail = indSeqHead + opt.lstmHist - 1
                    end
                end
            end
            self.rnnRealUserDataEndsTest[#self.rnnRealUserDataEndsTest+1] = #self.rnnRealUserDataStatesTest     -- Set the end of the last user's record
            -- There are in total 15509 sequences if histLen is 3. 14707 if histLen is 5. 15108 if histLen is 4. 15911 if histLen is 2.
        end
    end


    ----------------------------------------------------------------------
    --- Prepare data for CNN models in training set
    ---
    self.cnnRealUserDataStates = {}
    self.cnnRealUserDataActs = {}
    self.cnnRealUserDataRewards = {}
    self.cnnRealUserDataStandardNLG = {}
    self.cnnRealUserDataStarts = {}
    self.cnnRealUserDataEnds = {}
    self.cnnRealUserDataPad = torch.Tensor(#self.realUserDataStartLines):fill(0)    -- indicating whether data has padding at head (should be padded)
    if string.sub(opt.uppModel, 1, 4) == 'cnn_' or string.sub(opt.uppModelUsp, 1, 4) == 'cnn_' then
        local indSeqHead = 1
        local indSeqTail = opt.lstmHist     -- for CNN-based player simulation models, we also use this lstmHist opt value as indicator of the history length in state representation (in CNN, it is number of frames at each time step)
        local indUserSeq = 1    -- user id ptr. Use this to get the tail of each trajectory
        while indSeqTail <= #self.realUserDataStates do
            if self.cnnRealUserDataPad[indUserSeq] < 1 then
                for padi = opt.lstmHist-1, 1, -1 do
                    self.cnnRealUserDataStates[#self.cnnRealUserDataStates + 1] = torch.Tensor(opt.lstmHist, self.userStateFeatureCnt)  -- input into TemporalConvolution is 2d or 3d tensor
                    self.cnnRealUserDataActs[#self.cnnRealUserDataActs + 1] = self.realUserDataActs[indSeqHead + opt.lstmHist - padi - 1]   -- act and reward were set to the value at last position in this
                    self.cnnRealUserDataRewards[#self.cnnRealUserDataRewards + 1] = self.realUserDataRewards[indSeqHead + opt.lstmHist-padi - 1]    -- time sequence
                    self.cnnRealUserDataStandardNLG[#self.cnnRealUserDataStandardNLG + 1] = self.realUserDataStandardNLG[indSeqHead + opt.lstmHist-padi - 1]
                    for i=1, padi do
                        self.cnnRealUserDataStates[#self.cnnRealUserDataStates][i]:fill(0)
                    end
                    for i=1, opt.lstmHist-padi do
                        self.cnnRealUserDataStates[#self.cnnRealUserDataStates][i+padi] = self.realUserDataStates[indSeqHead+i-1]
                    end
                    if padi == opt.lstmHist-1 then
                        self.cnnRealUserDataStarts[#self.cnnRealUserDataStarts+1] = #self.cnnRealUserDataStates     -- This is the start of a user's record -- This is duplicated. The value should be the same as realUserDataStartLines
                    end
                    if indSeqHead+(opt.lstmHist-padi)-1 == self.realUserDataEndLines[indUserSeq] then
                        self.cnnRealUserDataPad[indUserSeq] = 1
                        break   -- if padding tail is going to outrange this user record's tail, break
                    end
                end
                self.cnnRealUserDataPad[indUserSeq] = 1
            else
                if indSeqTail <= self.realUserDataEndLines[indUserSeq] then
                    self.cnnRealUserDataStates[#self.cnnRealUserDataStates + 1] = torch.Tensor(opt.lstmHist, self.userStateFeatureCnt)  -- input into TemporalConvolution is 2d or 3d tensor
                    self.cnnRealUserDataActs[#self.cnnRealUserDataActs + 1] = self.realUserDataActs[indSeqHead + opt.lstmHist - 1]
                    self.cnnRealUserDataRewards[#self.cnnRealUserDataRewards + 1] = self.realUserDataRewards[indSeqHead + opt.lstmHist - 1]
                    self.cnnRealUserDataStandardNLG[#self.cnnRealUserDataStandardNLG + 1] = self.realUserDataStandardNLG[indSeqHead + opt.lstmHist - 1]
                    for i=1, opt.lstmHist do
                        self.cnnRealUserDataStates[#self.cnnRealUserDataStates][i] = self.realUserDataStates[indSeqHead+i-1]
                    end
                    indSeqHead = indSeqHead + 1
                    indSeqTail = indSeqTail + 1
                else
                    self.cnnRealUserDataEnds[#self.cnnRealUserDataEnds+1] = #self.cnnRealUserDataStates     -- This is the end of a user's record
                    indUserSeq = indUserSeq + 1 -- next user's records
                    indSeqHead = self.realUserDataStartLines[indUserSeq]
                    indSeqTail = indSeqHead + opt.lstmHist - 1
                end
            end
        end
        self.cnnRealUserDataEnds[#self.cnnRealUserDataEnds+1] = #self.cnnRealUserDataStates     -- Set the end of the last user's record
        -- There are in total 15509 sequences if histLen is 3. 14707 if histLen is 5. 15108 if histLen is 4. 15911 if histLen is 2.
    end

    ----------------------------------------------------------------------
    --- Prepare data for CNN models in test/train_validation set
    ---
    self.cnnRealUserDataStatesTest = {}
    self.cnnRealUserDataActsTest = {}
    self.cnnRealUserDataRewardsTest = {}
    self.cnnRealUserDataStandardNLGTest = {}
    self.cnnRealUserDataStartsTest = {}
    self.cnnRealUserDataEndsTest = {}
    self.cnnRealUserDataPadTest = torch.Tensor(#self.realUserDataStartLinesTest):fill(0)    -- indicating whether data has padding at head (should be padded)
    if self.opt.ciuTType == 'train' or self.opt.ciuTType == 'train_tr' then
        if string.sub(opt.uppModel, 1, 4) == 'cnn_' then
            local indSeqHead = 1
            local indSeqTail = opt.lstmHist
            local indUserSeq = 1    -- user id ptr. Use this to get the tail of each trajectory
            while indSeqTail <= #self.realUserDataStatesTest do
                if self.cnnRealUserDataPadTest[indUserSeq] < 1 then
                    for padi = opt.lstmHist-1, 1, -1 do
                        self.cnnRealUserDataStatesTest[#self.cnnRealUserDataStatesTest + 1] = torch.Tensor(opt.lstmHist, self.userStateFeatureCnt)  -- input into TemporalConvolution is 2d or 3d tensor
                        self.cnnRealUserDataActsTest[#self.cnnRealUserDataActsTest + 1] = self.realUserDataActsTest[indSeqHead + opt.lstmHist - padi - 1]   -- act and reward were set to the value at last position in this
                        self.cnnRealUserDataRewardsTest[#self.cnnRealUserDataRewardsTest + 1] = self.realUserDataRewardsTest[indSeqHead + opt.lstmHist-padi - 1]    -- time sequence
                        self.cnnRealUserDataStandardNLGTest[#self.cnnRealUserDataStandardNLGTest + 1] = self.realUserDataStandardNLGTest[indSeqHead + opt.lstmHist-padi - 1]    -- time sequence
                        for i=1, padi do
                            self.cnnRealUserDataStatesTest[#self.cnnRealUserDataStatesTest][i]:fill(0)
                        end
                        for i=1, opt.lstmHist-padi do
                            self.cnnRealUserDataStatesTest[#self.cnnRealUserDataStatesTest][i+padi] = self.realUserDataStatesTest[indSeqHead+i-1]
                        end
                        if padi == opt.lstmHist-1 then
                            self.cnnRealUserDataStartsTest[#self.cnnRealUserDataStartsTest+1] = #self.cnnRealUserDataStatesTest     -- This is the start of a user's record
                        end
                        if indSeqHead+(opt.lstmHist-padi)-1 == self.realUserDataEndLinesTest[indUserSeq] then
                            self.cnnRealUserDataPadTest[indUserSeq] = 1
                            break   -- if padding tail is going to outrange this user record's tail, break
                        end
                    end
                    self.cnnRealUserDataPadTest[indUserSeq] = 1
                else
                    if indSeqTail <= self.realUserDataEndLinesTest[indUserSeq] then
                        self.cnnRealUserDataStatesTest[#self.cnnRealUserDataStatesTest + 1] = torch.Tensor(opt.lstmHist, self.userStateFeatureCnt)  -- input into TemporalConvolution is 2d or 3d tensor
                        self.cnnRealUserDataActsTest[#self.cnnRealUserDataActsTest + 1] = self.realUserDataActsTest[indSeqHead + opt.lstmHist - 1]
                        self.cnnRealUserDataRewardsTest[#self.cnnRealUserDataRewardsTest + 1] = self.realUserDataRewardsTest[indSeqHead + opt.lstmHist - 1]
                        self.cnnRealUserDataStandardNLGTest[#self.cnnRealUserDataStandardNLGTest + 1] = self.realUserDataStandardNLGTest[indSeqHead + opt.lstmHist - 1]
                        for i=1, opt.lstmHist do
                            self.cnnRealUserDataStatesTest[#self.cnnRealUserDataStatesTest][i] = self.realUserDataStatesTest[indSeqHead+i-1]
                        end
                        indSeqHead = indSeqHead + 1
                        indSeqTail = indSeqTail + 1
                    else
                        self.cnnRealUserDataEndsTest[#self.cnnRealUserDataEndsTest+1] = #self.cnnRealUserDataStatesTest     -- This is the end of a user's record
                        indUserSeq = indUserSeq + 1 -- next user's records
                        indSeqHead = self.realUserDataStartLinesTest[indUserSeq]
                        indSeqTail = indSeqHead + opt.lstmHist - 1
                    end
                end
            end
            self.cnnRealUserDataEndsTest[#self.cnnRealUserDataEndsTest+1] = #self.cnnRealUserDataStatesTest     -- Set the end of the last user's record
            -- There are in total 15509 sequences if histLen is 3. 14707 if histLen is 5. 15108 if histLen is 4. 15911 if histLen is 2.
        end
    end


    --- The following tensors are used to calculated Pearson's correlations between state features
    --- in self.realUserDataStates
    self.featSqre = torch.Tensor(self.userStateFeatureCnt):zero()
    self.featCrossSqre = torch.Tensor(self.userStateFeatureCnt, self.userStateFeatureCnt):zero()
    self.featMean = torch.Tensor(self.userStateFeatureCnt):zero()   -- feature mean in realUserDataStates
    self.featStdDev = torch.Tensor(self.userStateFeatureCnt):zero() -- standard deviation of each feature in realUserDataStates
    self.featCorreTable = torch.Tensor(self.userStateFeatureCnt, self.userStateFeatureCnt):zero()   -- one table stores features' correlation with each other
    self.featCorreTabSortRes = nil  -- sorted feature absolute correlations (including all features, player acts, planner acts, player characteristics)
    self.featCorreTabSortRank = nil -- rank of sorted feature absolute correlations (including all features, player acts, planner acts, player characteristics)
    self.featOfActCorreTable = nil  -- a table only stores player action feature correlations
    self.featOfActCorreTabSortRes = nil     -- sorted feature absolute correlations (including only player acts)
    self.featOfActCorreTabRank = nil    -- rank of sorted feature absolute correlations (including only player acts)
    self.featOfActCorreAbsTabCumsum = nil    -- get the absolute values of action correlation coefficients, and get the cumsum of each row
    self:_PearsonCorrCalc() -- Calculate a-squared, b-squared and a*b, that are all required in Pearson's correlation calculation

    --- The following tensors are used to record statistics of actions observed in training set
    self.actCntTotal = torch.Tensor(self.CIFr.usrActInd_end):fill(1e-5)
    self.actFreqTotal = torch.Tensor(self.CIFr.usrActInd_end):fill(1e-5)
    self.actFreqSortResCum = nil    -- action frequency cumsum after descending sort
    self.actFreqSortRank = nil  -- action frequency rank after descending sort
    self.actFreqSigmoid = nil   -- action frequency after through a sigmoid function. This is used in purpose of flatting action distribution
    self.actFreqSigmoidCum = nil    -- cumsum of the above self.actFreqSigmoid
    self.priorActStatThres = 20     -- This is the constant threshold parameter indicating for how long time step from currect time step the prior action stats we're going to calculate
    self.actCntPriorStep = torch.Tensor(self.CIFr.usrActInd_end, self.priorActStatThres, self.CIFr.usrActInd_end):fill(1e-5)
    self.actFreqPriorStep = torch.Tensor(self.CIFr.usrActInd_end, self.priorActStatThres, self.CIFr.usrActInd_end):fill(1e-5)
    self.actRankPriorStep = torch.Tensor(self.CIFr.usrActInd_end, self.priorActStatThres, self.CIFr.usrActInd_end):fill(1)
    self.actSigmoidDistPriorStep = torch.Tensor(self.CIFr.usrActInd_end, self.priorActStatThres, self.CIFr.usrActInd_end):fill(1e-5)
    self.actSigmoidDistPriorStepCumsum = nil
    self:_actionFreqCalc()  -- Calculate prior action appearance frequency

    collectgarbage()
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

--- Our preprocessing can do re-scaling or normalization
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

    local _stateFeatMean, _stateFeatStd, _stateFeatRescale
    if obvUserData:squeeze():dim() == 1 then
        -- If input Tensor obvUerDatais 1-d (only one data point)
        _stateFeatMean = self.stateFeatureMeanEachFeature
        _stateFeatStd = self.stateFeatureStdEachFeature
        _stateFeatRescale = self.stateFeatureRescaleFactor
    elseif obvUserData:dim() == 2 then
        -- If input tensor is 2-d (each row contains one data point)
        -- Duplicate mean and std tensors and make them fit the dim of input tensor
        -- We assume the 1st dim is batch index
        -- Right now, RNN model uses this format
        -- non-sequential model in batch also uses this format
        _stateFeatMean = torch.repeatTensor(self.stateFeatureMeanEachFeature, obvUserData:size(1), 1)
        _stateFeatStd = torch.repeatTensor(self.stateFeatureStdEachFeature, obvUserData:size(1), 1)
        _stateFeatRescale = torch.repeatTensor(self.stateFeatureRescaleFactor, obvUserData:size(1), 1)
    elseif obvUserData:dim() == 3 then
        -- Right now, CNN model uses this format
        -- Input tensor is 3-d. 1st dim is batch index,
        -- 2nd dim is frame index (history index, with total length of opt.lstmHist)
        -- 3rd dim is feature index
        _stateFeatMean = torch.repeatTensor(self.stateFeatureMeanEachFeature, obvUserData:size(1), obvUserData:size(2), 1)
        _stateFeatStd = torch.repeatTensor(self.stateFeatureStdEachFeature, obvUserData:size(1), obvUserData:size(2), 1)
        _stateFeatRescale = torch.repeatTensor(self.stateFeatureRescaleFactor, obvUserData:size(1), obvUserData:size(2), 1)
    else
        print('Size of input data does not fit in preprocessing')
        os.exit()
    end

    if ppType == 'rsc' then
        return torch.cdiv(obvUserData, _stateFeatRescale)
    elseif ppType == 'std' then
        local subMean = torch.add(obvUserData, -1, _stateFeatMean)
        return torch.cdiv(subMean, _stateFeatStd)
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
function CIUserSimulator:_PearsonCorrCalc()
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

    -- Record mean for each feature in self.realUserDataStates. This is not relavent to Pearson's correlation calculation
    self.featMean = stateMean

    local diffWithMean = torch.Tensor(#self.realUserDataStates, stateDim):zero()
    for i=1, #self.realUserDataStates do
        diffWithMean[i] = self.realUserDataStates[i] - stateMean
    end

    -- get squared values of all features in all standardized data points
    local allEleSqr = torch.pow(diffWithMean, 2)

    -- Get sum over all data points, for each feature
    self.featSqre = torch.sum(allEleSqr, 1):squeeze() -- squeeze() is needed to make it 1-D tensor

    -- a*b for each two features in the state representation. self.featCrossSqre has been constructed in _init()
    -- Time to calculate the a*b. Consider to use tensor:sub, or narrow, or select functions.
    for j=1, stateDim do
        for k=j, stateDim do
            self.featCrossSqre[j][k] = diffWithMean:select(2, j) * diffWithMean:select(2, k)    -- dot product of two standardized columns
            self.featCrossSqre[k][j] = self.featCrossSqre[j][k]
        end
    end

    -- Calculate the standard deviation of each feature in self.realUserDataStates. This is not used in Pearson's correlation calculation
    -- So, by-product
    self.featStdDev = torch.sqrt(self.featSqre / (#self.realUserDataStates-1))

    -- Calculate Pearson's correlation coefficient of each pair of features in state representation
    for i=1, self.userStateFeatureCnt do
        for j=i, self.userStateFeatureCnt do
            self.featCorreTable[i][j] = self.featCrossSqre[i][j] / math.sqrt(self.featSqre[i]) / math.sqrt(self.featSqre[j])
            self.featCorreTable[j][i] = self.featCorreTable[i][j]
        end
    end

    -- Sort the abs of Pearson's correlation along each line(feature)
    -- Remember the sorted values are abs of Pearson's correlation here
    self.featCorreTabSortRes, self.featCorreTabSortRank = torch.sort(torch.abs(self.featCorreTable), 2)

    -- Get the correlation table for only player action features
    -- This table is a one part of the self.featCorreTable
    self.featOfActCorreTable = self.featCorreTable[{{1, self.CIFr.usrActInd_end-1}, {1, self.CIFr.usrActInd_end-1}}]:clone()
    -- Sort the abs of Pearson's correlation along each line in between only player action features, ascending order
    self.featOfActCorreTabSortRes, self.featOfActCorreTabRank = torch.sort(torch.abs(self.featOfActCorreTable), 2)
    self.featOfActCorreAbsTabCumsum = torch.cumsum(torch.abs(self.featOfActCorreTable), 2)
    local _lastCumsumItemTensor = self.featOfActCorreAbsTabCumsum[{{}, {self.CIFr.usrActInd_end-1}}]    -- this is a 2-d (self.CIFr.usrActInd_end-1 * 1) matrix
    self.featOfActCorreAbsTabCumsum:cdiv(torch.repeatTensor(_lastCumsumItemTensor, 1, self.CIFr.usrActInd_end-1):add(1e-5))    -- standardization

end

-- Return the Pearson's correlation between two features in self.realUserDataStates
function CIUserSimulator:PearsonCorrelationOfTwo(feat1, feat2)
    assert(feat1>=1 and feat1 <=self.userStateFeatureCnt)
    assert(feat2>=1 and feat2 <=self.userStateFeatureCnt)
    --return self.featCrossSqre[feat1][feat2] / math.sqrt(self.featSqre[feat1]) / math.sqrt(self.featSqre[feat2])
    return self.featCorreTable[feat1][feat2]
end

-- Do statistics of action frequency
function CIUserSimulator:_actionFreqCalc()
    for i=1, #self.realUserDataActs do
        self.actCntTotal[self.realUserDataActs[i]] = self.actCntTotal[self.realUserDataActs[i]] + 1
    end
    self.actFreqTotal = torch.div(self.actCntTotal, #self.realUserDataActs)
    -- In fact it is not necessary to sort act frequency for the sampling purpose. It just offers a convenient way to check to sorting result.
    -- And the sort is only invoked once. So I can leave it here right now.
    self.actFreqSortResCum, self.actFreqSortRank = torch.sort(self.actFreqTotal[{{1, self.CIFr.usrActInd_end-1}}], true)  -- descending order, only rank first 14 actions (not including game-ending action)
    self.actFreqSortResCum = torch.cumsum(self.actFreqSortResCum)   -- Get the cumsum of player action frequency
    self.actFreqSortResCum:div(self.actFreqSortResCum[self.CIFr.usrActInd_end-1])   -- standardization
    -- Try this, calculate the sigmoid distribution of actions
    self.actFreqSigmoid = torch.div(self.actCntTotal, #self.realUserDataStartLines) -- This is action frequency per sequence
    self.actFreqSigmoid = torch.sigmoid(self.actFreqSigmoid - torch.Tensor(self.actFreqSigmoid:size()):fill(torch.mean(self.actFreqSigmoid)))    -- calculate the sigmoid distribution of action freq per sequence minus Mean. This is a hyper-param
    self.actFreqSigmoidCum = torch.cumsum(self.actFreqSigmoid[{{1, self.CIFr.usrActInd_end-1}}])    -- cumsum of all action sigmoid distribution except for game-ending action
    self.actFreqSigmoidCum:div(self.actFreqSigmoidCum[self.CIFr.usrActInd_end-1])   -- standardization. Attention: this sigmoid distribution does not include game-ending action

    for i=1, #self.realUserDataActs-1 do
        if self.realUserDataActs[i] ~= self.CIFr.usrActInd_end then
            for j=1, self.priorActStatThres do
                if self.realUserDataActs[i+j-1] ~= self.CIFr.usrActInd_end then   -- self.CIFr.usrActInd_end is the last action in the action set, also the ending action
                    -- Count action appearance of current action at step i as a prior action j steps prior to the future step
                    self.actCntPriorStep[self.realUserDataActs[i+j]][j][self.realUserDataActs[i]] = self.actCntPriorStep[self.realUserDataActs[i+j]][j][self.realUserDataActs[i]] + 1
                else
                    break   -- We need to guarantee the ending action is the last action in each interaction sequence
                end
            end
        end
    end

    -- normalize prior action counting along the distance dimension
    self.actSigmoidDistPriorStep = self.actCntPriorStep:clone()
    local priorActAlongDisMean = torch.mean(self.actCntPriorStep, 2)    -- the 2nd dim is distance between two actions
    local priorActAlongDisStd = torch.std(self.actCntPriorStep, 2):add(1e-5)    -- Attention: this add(1e-5) will avoid divided by 0 problem in the standardization process (cdiv() below)
    for ite=1, self.actSigmoidDistPriorStep:size(2) do
        self.actSigmoidDistPriorStep[{{}, {ite}, {}}]:csub(priorActAlongDisMean)
        self.actSigmoidDistPriorStep[{{}, {ite}, {}}]:cdiv(priorActAlongDisStd)
        self.actSigmoidDistPriorStep[{{}, {ite}, {}}]:mul(2)    -- Try to broaden the threshold a little bit
    end
    self.actSigmoidDistPriorStep:sigmoid()  -- Get the sigmoid distribution
    -- Attention: I've seen that several rows along the 2nd dimension have all 0 values in this tensor. Specifically
    -- it means no such action pair appeared in the corpus. I would guess it will introduce problems in sampling. So,
    -- I'll add very small positive constants to all elements in this tensor
    self.actSigmoidDistPriorStep:add(1e-5)
    self.actSigmoidDistPriorStepCumsum = torch.cumsum(self.actSigmoidDistPriorStep, 2)

    local priorActSum = torch.cumsum(self.actCntPriorStep, 3)
    for i=1, self.CIFr.usrActInd_end do
        for j=1, self.priorActStatThres do
            for k=1, self.CIFr.usrActInd_end-1 do
                priorActSum[i][j][k] = priorActSum[i][j][self.CIFr.usrActInd_end]
            end
        end
    end
    priorActSum:add(1e-5)   -- avoid dividing 0
    self.actFreqPriorStep = torch.cdiv(self.actCntPriorStep, priorActSum)

    _, self.actRankPriorStep = torch.sort(self.actFreqPriorStep, 3, true)   -- descending order

end

function CIUserSimulator:calcOneStepActPertCounting(p_act_cnt, p_act_ind, _dual_act_change_prob)
    if self.actFreqTotal[p_act_ind] < self.actFreqTotal[self.CIFr.usrActInd_end] and torch.uniform() < 0.75 then -- as a ref, end-game has actFreqTotal of 0.0245
        -- some actions are rarely adopted.
        p_act_cnt = 0
    elseif self.actFreqTotal[p_act_ind] <= self.actFreqTotal[self.CIFr.usrActInd_end] * 2 then
        local _samSeed = torch.uniform()
        if _samSeed < _dual_act_change_prob then
            p_act_cnt = p_act_cnt * 2
        elseif _samSeed > 0.75 then
            p_act_cnt = 0
        end
    elseif self.actFreqTotal[p_act_ind] <= self.actFreqTotal[self.CIFr.usrActInd_end] * 4 then
        if torch.uniform() < _dual_act_change_prob then
            p_act_cnt = p_act_cnt * 2
        end
        if torch.uniform() < _dual_act_change_prob/4 then
            p_act_cnt = p_act_cnt/math.abs(p_act_cnt) * 3
        end
    else
        if torch.uniform() < _dual_act_change_prob then
            p_act_cnt = p_act_cnt * 2
        end
        if torch.uniform() < _dual_act_change_prob/3 then
            p_act_cnt = p_act_cnt/math.abs(p_act_cnt) * 3
        end
    end

    return p_act_cnt
end

--- Augment data in training corpus using the proteties we extract here
--- In this version, action and score are sent together as input param, so generated
--- augmented data points will have corresponding action and score labels together
function CIUserSimulator:UserSimActDataAugment(input, output_act, output_score_classifier, output_score_regressor, uppModelForm)
    if string.sub(uppModelForm, 1, 4) == 'rnn_' then
        -- If the player simulation model is an RNN model, which means the model is a sequencer,
        -- and requires input to be a table of tensors in which the 1st dim (of the table) is time step
        for j = 1, self.opt.lstmHist do
            input[j]:resize(self.opt.batchSize * 2, self.userStateFeatureCnt)
            output_act[j]:resize(self.opt.batchSize * 2)
            output_score_classifier[j]:resize(self.opt.batchSize * 2)
            output_score_regressor[j]:resize(self.opt.batchSize * 2)
        end

        for i=1, self.opt.batchSize do
            -- clone from original
            for j=1, self.opt.lstmHist do
                input[j][self.opt.batchSize+i] = input[j][i]
                output_act[j][self.opt.batchSize+i] = output_act[j][i]
                output_score_classifier[j][self.opt.batchSize+i] = output_score_classifier[j][i]
                output_score_regressor[j][self.opt.batchSize+i] = output_score_regressor[j][i]
                -- for output_score_classifier and output_score_regressor, we only need to double-sized the tensors, and does not need to make any changes.
                -- This is because the current data augmentation, or perturbation is based on inserting/deleting actions
                -- in the interaction sequence. In our CI dataset, the score label is duplicated for each time step in
                -- one player's interaction. So, we just double-size it here and it should correspond to augmented data points.
            end

            local actStepCntTotal = torch.cumsum(input[self.opt.lstmHist][self.opt.batchSize+i])[self.CIFr.usrActInd_end-1]    -- the counting of all actions the player took till now
            local freqActPertProb
            if actStepCntTotal < 3 then
                freqActPertProb = {}
            elseif actStepCntTotal < 6 then
                -- For action 4-6, the standard deviations are not that large, so try to perturb slightly
                freqActPertProb = {0.3}--{0.3}
            elseif actStepCntTotal <= 20 then
                freqActPertProb = {0.7, 0.5, 0.3, 0.2, 0.1}--{0.7, 0.35}
            elseif actStepCntTotal <= 35 then
                freqActPertProb = {0.8, 0.7, 0.6, 0.45, 0.25, 0.1}--{0.8, 0.5, 0.3}
            else
                freqActPertProb = {0.9, 0.8, 0.7, 0.5, 0.35, 0.25, 0.1}--{0.9, 0.7, 0.35}
            end

            assert(self.priorActStatThres >= self.opt.lstmHist)     -- I think in ideal case self.priorActStatThres should be much larger than opt.lstmHist
            local _pertActCandidateList = self.actFreqSigmoidCum    -- It is set up in this way bcz we cannot sample actions w.r.t correlation for game-ending action
            if output_act[self.opt.lstmHist][self.opt.batchSize+i] ~= self.CIFr.usrActInd_end then  -- if current action is not ending action, then use correlation, otherwise use action freq
                _pertActCandidateList = self.featOfActCorreAbsTabCumsum[output_act[self.opt.lstmHist][self.opt.batchSize+i]]
            end

            for k=1, #freqActPertProb do
                assert(actStepCntTotal >= 2 )
                if torch.uniform() < freqActPertProb[k] then
                    -- Sample an perturbed action according to correlation or action frequency
                    local freqActSmpSeed = torch.uniform()
                    for pai=1, self.featOfActCorreAbsTabCumsum:size(2) do
                        if freqActSmpSeed <= _pertActCandidateList[pai] then
                            -- This is designed differently from the correlation based perturbation method for non-rnn, non-cnn data augmentation
                            -- So, each time we only perturb one action, and this action can be repeatedly perturbed
                            -- If the program enters here, that means the action "pai" is the chosen one to be perturbed
                            -- Right now, we determine where this action perturbation should be.
                            -- The following cumsum calculates the cumsum of appearances of action pai being priorly k steps to action output_act[j][self.opt.batchSize+i]

                            if torch.uniform() < 0.5 then
                                --- Add an extra action into this sequence
                                local priorActCumsum = self.actSigmoidDistPriorStepCumsum[{ output_act[self.opt.lstmHist][self.opt.batchSize+i], {}, pai}]:clone() -- clone() should be necessary, since we change its value later
                                -- valActSampDist is the valid number of time steps we can count backwards to add an extra player action.
                                -- Attention: actStepCntTotal is 1 step less than the current time, bcz it is calculated from input features, which contain cumsum of prior actions
                                local valActSampDist = math.min(actStepCntTotal, self.priorActStatThres)
                                priorActCumsum:div(priorActCumsum[valActSampDist])  -- standardization

                                -- sample a position to add an action
                                local pertActPosPriorInRnn = 0  -- The backward distance of the perturbed action w.r.t current action (output_act[lstmHist][opt.batchSize+i])
                                local actPosSampSeed = torch.uniform()

                                for sp=1, valActSampDist do
                                    if actPosSampSeed <= priorActCumsum[sp] then
                                        pertActPosPriorInRnn = sp    -- This is a relative position, the backward distance w.r.t current position
                                        break
                                    end
                                end

                                -- Now, we try to add extra actions
                                if pertActPosPriorInRnn < self.opt.lstmHist then
                                    -- For actions prior to the newly added action, move them front
                                    for frm=1, self.opt.lstmHist - pertActPosPriorInRnn - 1 do
                                        input[frm][self.opt.batchSize+i] = input[frm+1][self.opt.batchSize+i]
                                        output_act[frm][self.opt.batchSize+i] = output_act[frm+1][self.opt.batchSize+i]
                                    end

                                    input[self.opt.lstmHist - pertActPosPriorInRnn][self.opt.batchSize+i] = input[self.opt.lstmHist - pertActPosPriorInRnn + 1][self.opt.batchSize+i]
                                    output_act[self.opt.lstmHist - pertActPosPriorInRnn][self.opt.batchSize+i] = pai
                                    for brm=self.opt.lstmHist - pertActPosPriorInRnn + 1, self.opt.lstmHist do
                                        input[brm][self.opt.batchSize+i][pai] = input[brm][self.opt.batchSize+i][pai] + 1
                                    end
                                else
                                    -- the newly added action should be long ago, longer than the input lstm sequence length
                                    for brm=1, self.opt.lstmHist do
                                        input[brm][self.opt.batchSize+i][pai] = input[brm][self.opt.batchSize+i][pai] + 1
                                    end
                                end
                            else
                                --- delete an existed action (if possible) in this sequence
                                local _delActInSeqWithinSeq = function ()
                                    -- This function is used to try to delete an action from a RNN sequence in which the deleted action should appear in the sequence
                                    local pActAppPos = {}
                                    for _paap=1, math.min(actStepCntTotal, self.opt.lstmHist-1) do
                                        if output_act[self.opt.lstmHist - _paap][self.opt.batchSize+i] == pai then
                                            pActAppPos[#pActAppPos+1] = _paap   -- _paap is a distance from lastest action to the sampled action
                                        end
                                    end
                                    -- If there are actions "pai" in this sequence that can be deleted, then try to find this action with a proper place (if there are multiple "pai"s)
                                    if #pActAppPos > 0 then
                                        local pActSigmDistT = torch.Tensor(#pActAppPos)
                                        for _obvAct=1, #pActAppPos do
                                            pActSigmDistT[_obvAct] = self.actSigmoidDistPriorStep[{ output_act[self.opt.lstmHist][self.opt.batchSize+i], pActAppPos[_obvAct], pai}]
                                        end
                                        pActSigmDistT:add(1e-5)     -- avoid divide by zero
                                        pActSigmDistT:cumsum()
                                        pActSigmDistT:div(pActSigmDistT[#pActAppPos])   -- standardization
                                        -- Sample that position of action "pai" to delete
                                        local _dltSampSeed = torch.uniform()
                                        for dltPos=1, #pActAppPos do
                                            if _dltSampSeed <= pActSigmDistT[dltPos] then
                                                -- delete action "pai" at this sampled position
                                                for fra=self.opt.lstmHist-pActAppPos[dltPos], 2, -1 do
                                                    input[fra][self.opt.batchSize+i] = input[fra-1][self.opt.batchSize+i]
                                                    output_act[fra][self.opt.batchSize+i] = output_act[fra-1][self.opt.batchSize+i]
                                                end
                                                input[1][self.opt.batchSize+i]:zero()
                                                for bra=self.opt.lstmHist-pActAppPos[dltPos] + 1, self.opt.lstmHist do
                                                    input[bra][self.opt.batchSize+i][pai] = input[bra][self.opt.batchSize+i][pai] - 1
                                                    if input[bra][self.opt.batchSize+i][pai] < 0 then
                                                        input[bra][self.opt.batchSize+i][pai] = 0
                                                    end
                                                end
                                                -- Done from action deletion in a sequence, in which case the sequence is shorter than opt.lstmHist. Break out of for-loop
                                                break
                                            end
                                        end
                                    end
                                end -- end of the function _delActInSeqWithinSeq definition

                                if actStepCntTotal < self.opt.lstmHist then
                                    ---- In this case, all previous actions are in the rnn sequence representation
                                    _delActInSeqWithinSeq()
                                else
                                    -- In this case, there were prior actions (at so early time) not recorded in the output_act list
                                    -- We first tell is the deleted "pai" action should be in a short range (in range of opt.lstmHist),
                                    -- in which case we need to sample that action in the squence we've had, or the deleted "pai" action
                                    -- is in a long range, which means it does not appear in the state representation, and we just delete
                                    -- that action counting
                                    local priorActCumsum = self.actSigmoidDistPriorStepCumsum[{ output_act[self.opt.lstmHist][self.opt.batchSize+i], {}, pai}]:clone() -- clone() should be necessary, since we change its value later
                                    -- valActSampDist is the valid number of time steps we can count backwards to delete an existed player action.
                                    -- Attention: actStepCntTotal is 1 step less than the current time, bcz it is calculated from input features, which contain cumsum of prior actions
                                    local valActSampDist = math.min(actStepCntTotal, self.priorActStatThres)
                                    priorActCumsum:div(priorActCumsum[valActSampDist])  -- standardization
                                    local _rndActPosRangeSeed = torch.uniform()
                                    if _rndActPosRangeSeed <= priorActCumsum[self.opt.lstmHist-1] then
                                        -- This case is that the deleted action should appear in the current state representation (distance from lastest action is smaller than opt.lstmHist)
                                        -- Then what we do here is pretty similar to the action deletion up there (if actStepCntTotal < self.opt.lstmHist)
                                        _delActInSeqWithinSeq()
                                    else
                                        -- This case is that the deleted action should NOT appear in the current rnn state sequence representation (the deleted action is not recorded in the input sequence)
                                        for _bapit=1, self.opt.lstmHist do
                                            input[_bapit][self.opt.batchSize+i][pai] = input[_bapit][self.opt.batchSize+i][pai] - 1
                                            if input[_bapit][self.opt.batchSize+i][pai] < 0 then
                                                input[_bapit][self.opt.batchSize+i][pai] = 0
                                            end
                                        end
                                    end

                                end
                            end

                            --- We've found an action to perturb its counting, so break from the for-loop
                            break
                        end
                    end
                end
            end
        end

    elseif string.sub(uppModelForm, 1, 4) == 'cnn_' then
        -- If this player simulation model is in CNN form, meaning its input may contain multiple
        -- frames of features, and action prediction output_act only contains one value. For data augmentation,
        -- the data is processed similar as it is for RNN models
        assert(input:dim()==3 and output_act:dim()==1 and output_score_classifier:dim()==1 and output_score_regressor:dim()==1, 'Input for CNN models in player simulator should be 3d')
        input:resize(self.opt.batchSize * 2, input:size(2), input:size(3))
        output_act:resize(self.opt.batchSize * 2)
        output_score_classifier:resize(self.opt.batchSize * 2)
        output_score_regressor:resize(self.opt.batchSize * 2)

        for i=1, self.opt.batchSize do
            -- clone from original
            input[self.opt.batchSize+i] = input[i]
            output_act[self.opt.batchSize+i] = output_act[i]
            output_score_classifier[self.opt.batchSize+i] = output_score_classifier[i]
            output_score_regressor[self.opt.batchSize+i] = output_score_regressor[i]

            local actStepCntTotal = torch.cumsum(input[self.opt.batchSize+i][self.opt.lstmHist])[self.CIFr.usrActInd_end-1]    -- the counting of all actions the player took till now
            local freqActPertProb
            if actStepCntTotal < 3 then
                freqActPertProb = {}
            elseif actStepCntTotal < 6 then
                -- For action 4-6, the standard deviations are not that large, so try to perturb slightly
                freqActPertProb = {0.3}--{0.3}
            elseif actStepCntTotal <= 20 then
                freqActPertProb = {0.7, 0.5, 0.3, 0.2, 0.1}--{0.7, 0.35}
            elseif actStepCntTotal <= 35 then
                freqActPertProb = {0.8, 0.7, 0.6, 0.45, 0.25, 0.1}--{0.8, 0.5, 0.3}
            else
                freqActPertProb = {0.9, 0.8, 0.7, 0.5, 0.35, 0.25, 0.1}--{0.9, 0.7, 0.35}
            end

            assert(self.priorActStatThres >= self.opt.lstmHist)     -- I think in ideal case self.priorActStatThres should be much larger than opt.lstmHist
            local _pertActCandidateList = self.actFreqSigmoidCum    -- It is set up in this way bcz we cannot sample actions w.r.t correlation for game-ending action
            if output_act[self.opt.batchSize+i] ~= self.CIFr.usrActInd_end then  -- if current action is not ending action, then use correlation, otherwise use action freq
                _pertActCandidateList = self.featOfActCorreAbsTabCumsum[output_act[self.opt.batchSize+i]]
            end

            for k=1, #freqActPertProb do
                assert(actStepCntTotal >= 2 )
                if torch.uniform() < freqActPertProb[k] then
                    -- Sample an perturbed action according to correlation or action frequency
                    local freqActSmpSeed = torch.uniform()
                    for pai=1, self.featOfActCorreAbsTabCumsum:size(2) do --self.actFreqSigmoidCum:size()[1] do
                        if freqActSmpSeed <= _pertActCandidateList[pai] then    --self.actFreqSigmoidCum[pai] then
                            -- This is designed differently from the correlation based perturbation method for mlp data augmentation
                            -- So, each time we only perturb one action, and this action can be repeatedly perturbed
                            -- If the program enters here, that means the action "pai" is the chosen one to be perturbed
                            -- Right now, we determine where this action perturbation should be.
                            -- The following cumsum calculates the cumsum of appearances of action pai being priorly k steps to action output_act[self.opt.batchSize+i]

                            if torch.uniform() < 0.5 then
                                --- Add an extra action into this sequence
                                local priorActCumsum = self.actSigmoidDistPriorStepCumsum[{ output_act[self.opt.batchSize+i], {}, pai}]:clone() -- clone() should be necessary, since we change its value later
                                -- valActSampDist is the valid number of time steps we can count backwards to add an extra player action.
                                -- Attention: actStepCntTotal is 1 step less than the current time, bcz it is calculated from input features, which contain cumsum of prior actions
                                local valActSampDist = math.min(actStepCntTotal, self.priorActStatThres)
                                priorActCumsum:div(priorActCumsum[valActSampDist])  -- standardization

                                -- sample a position to add an action
                                local pertActPosPriorInCnn = 0  -- The backward distance of the perturbed action w.r.t current action (output_act[lstmHist][opt.batchSize+i])
                                local actPosSampSeed = torch.uniform()

                                for sp=1, valActSampDist do
                                    if actPosSampSeed <= priorActCumsum[sp] then
                                        pertActPosPriorInCnn = sp    -- This is a relative position, the backward distance w.r.t current position
                                        break
                                    end
                                end

                                -- Now, we try to add extra actions
                                if pertActPosPriorInCnn < self.opt.lstmHist then
                                    -- For actions prior to the newly added action, move them front
                                    for frm=1, self.opt.lstmHist - pertActPosPriorInCnn - 1 do
                                        input[self.opt.batchSize+i][frm] = input[self.opt.batchSize+i][frm+1]
                                    end

                                    input[self.opt.batchSize+i][self.opt.lstmHist - pertActPosPriorInCnn] = input[self.opt.batchSize+i][self.opt.lstmHist - pertActPosPriorInCnn + 1]
                                    for brm=self.opt.lstmHist - pertActPosPriorInCnn + 1, self.opt.lstmHist do
                                        input[self.opt.batchSize+i][brm][pai] = input[self.opt.batchSize+i][brm][pai] + 1
                                    end
                                else
                                    -- the newly added action should be long ago, longer than the input lstm sequence length
                                    for brm=1, self.opt.lstmHist do
                                        input[self.opt.batchSize+i][brm][pai] = input[self.opt.batchSize+i][brm][pai] + 1
                                    end
                                end
                            else
                                --- delete an existed action (if possible) in this sequence
                                local _delActInSeqWithinSeq = function ()
                                    -- This function is used to try to delete an action from a RNN sequence in which the deleted action should appear in the sequence
                                    local pActAppPos = {}
                                    for _paap=1, math.min(actStepCntTotal, self.opt.lstmHist-1) do
                                        -- what is different from rnn data structure is that, cnn data only has one output_act, so we can only use input difference
                                        -- to check what action the player took from one time step ago
                                        if input[self.opt.batchSize+i][self.opt.lstmHist-_paap+1][pai] == input[self.opt.batchSize+i][self.opt.lstmHist-_paap][pai]+1 then
                                            pActAppPos[#pActAppPos+1] = _paap   -- _paap is a distance from lastest action to the sampled action
                                        end
                                    end
                                    -- If there are actions "pai" in this sequence that can be deleted, then try to find this action with a proper place (if there are multiple "pai"s)
                                    if #pActAppPos > 0 then
                                        local pActSigmDistT = torch.Tensor(#pActAppPos)
                                        for _obvAct=1, #pActAppPos do
                                            pActSigmDistT[_obvAct] = self.actSigmoidDistPriorStep[{ output_act[self.opt.batchSize+i], pActAppPos[_obvAct], pai}]
                                        end
                                        pActSigmDistT:add(1e-5)     -- avoid divide by zero
                                        pActSigmDistT:cumsum()
                                        pActSigmDistT:div(pActSigmDistT[#pActAppPos])   -- standardization
                                        -- Sample that position of action "pai" to delete
                                        local _dltSampSeed = torch.uniform()
                                        for dltPos=1, #pActAppPos do
                                            if _dltSampSeed <= pActSigmDistT[dltPos] then
                                                -- delete action "pai" at this sampled position
                                                for fra=self.opt.lstmHist-pActAppPos[dltPos], 2, -1 do
                                                    input[self.opt.batchSize+i][fra] = input[self.opt.batchSize+i][fra-1]
                                                end
                                                input[self.opt.batchSize+i][1]:zero()
                                                for bra=self.opt.lstmHist-pActAppPos[dltPos] + 1, self.opt.lstmHist do
                                                    input[self.opt.batchSize+i][bra][pai] = input[self.opt.batchSize+i][bra][pai] - 1
                                                    if input[self.opt.batchSize+i][bra][pai] < 0 then
                                                        input[self.opt.batchSize+i][bra][pai] = 0
                                                    end
                                                end
                                                -- Done from action deletion in a sequence, in which case the sequence is shorter than opt.lstmHist. Break out of for-loop
                                                break
                                            end
                                        end
                                    end
                                end -- end of the function _delActInSeqWithinSeq definition

                                if actStepCntTotal < self.opt.lstmHist then
                                    ---- In this case, all previous actions are in the rnn sequence representation
                                    _delActInSeqWithinSeq()
                                else
                                    -- In this case, there were prior actions (at so early time) not recorded in the output_act list
                                    -- We first tell is the deleted "pai" action should be in a short range (in range of opt.lstmHist),
                                    -- in which case we need to sample that action in the squence we've had, or the deleted "pai" action
                                    -- is in a long range, which means it does not appear in the state representation, and we just delete
                                    -- that action counting
                                    local priorActCumsum = self.actSigmoidDistPriorStepCumsum[{ output_act[self.opt.batchSize+i], {}, pai}]:clone() -- clone() should be necessary, since we change its value later
                                    -- valActSampDist is the valid number of time steps we can count backwards to delete an existed player action.
                                    -- Attention: actStepCntTotal is 1 step less than the current time, bcz it is calculated from input features, which contain cumsum of prior actions
                                    local valActSampDist = math.min(actStepCntTotal, self.priorActStatThres)
                                    priorActCumsum:div(priorActCumsum[valActSampDist])  -- standardization
                                    local _rndActPosRangeSeed = torch.uniform()
                                    if _rndActPosRangeSeed <= priorActCumsum[self.opt.lstmHist-1] then
                                        -- This case is that the deleted action should appear in the current state representation (distance from lastest action is smaller than opt.lstmHist)
                                        -- Then what we do here is pretty similar to the action deletion up there (if actStepCntTotal < self.opt.lstmHist)
                                        _delActInSeqWithinSeq()
                                    else
                                        -- This case is that the deleted action should NOT appear in the current rnn state sequence representation (the deleted action is not recorded in the input sequence)
                                        for _bapit=1, self.opt.lstmHist do
                                            input[self.opt.batchSize+i][_bapit][pai] = input[self.opt.batchSize+i][_bapit][pai] - 1
                                            if input[self.opt.batchSize+i][_bapit][pai] < 0 then
                                                input[self.opt.batchSize+i][_bapit][pai] = 0
                                            end
                                        end
                                    end

                                end
                            end

                            --- We've found an action to perturb its counting, so break from the for-loop
                            break
                        end
                    end
                end
            end
        end

    else
        -- If the model is non-RNN or non-CNN form, which means each input just contains feature values
        -- at the current time step. Right now, the strategy is using original data points and
        -- augmented data points in ratio of 1:1
        input:resize(self.opt.batchSize * 2, self.userStateFeatureCnt)
        output_act:resize(self.opt.batchSize * 2)
        output_score_classifier:resize(self.opt.batchSize * 2)
        output_score_regressor:resize(self.opt.batchSize * 2)

        -- Okay, try to utilize correlation in perturbing feature values
        for i=1, self.opt.batchSize do
            -- clone from original
            input[self.opt.batchSize+i] = input[i]
            output_act[self.opt.batchSize+i] = output_act[i]
            output_score_classifier[self.opt.batchSize+i] = output_score_classifier[i]
            output_score_regressor[self.opt.batchSize+i] = output_score_regressor[i]

            if output_act[i] ~= self.CIFr.usrActInd_end then -- and self.actFreqTotal[output_act[i]] < self.actFreqTotal[self.CIFr.usrActInd_end] * 5
                -- perturb feature values (action counting) according to correlation
                -- From the experiment we found that changing counting of highly correlated actions are helpful
                local correActPertProb  --{0.5, 0.3, 0.1} -- this set is good. MLP-bi act pred reaches to 33.5% high.
                -- The average sequence length is around 40. So, we should set correActPertProb accordingly
                local actStepCntTotal = torch.cumsum(input[self.opt.batchSize+i])[self.CIFr.usrActInd_end-1]    -- the counting of all actions the player took till now
                if actStepCntTotal <= 3 then
                    -- The standard deviations of the first 3 actions are small. So do not perturb data points here
                    correActPertProb = {}
                elseif actStepCntTotal <= 6 then
                    -- For action 4-6, the standard deviations are not that large, so try to perturb slightly
                    correActPertProb = {0.25, 0.2, 0.15}
                elseif actStepCntTotal <= 20 then
                    correActPertProb = {0.3, 0.3, 0.25, 0.2}
                elseif actStepCntTotal <= 35 then
                    correActPertProb = {0.3, 0.45, 0.4, 0.3, 0.15}
                else
                    correActPertProb = {0.5, 0.55, 0.45, 0.35, 0.2, 0.15}
                end
                for k=1, #correActPertProb do
                    if torch.uniform() < correActPertProb[k] then
                        local p_act_ind = self.featOfActCorreTabRank[output_act[i]][self.CIFr.usrActInd_end-k]
                        local p_act_cnt = 1
                        -- Half of the possibility to reduce counting
                        if torch.random(1,2) == 2 then
                            p_act_cnt = -1
                        end
                        -- If std is large, then make the case possible to change act count larger than 2
                        local _dual_act_change_prob = #correActPertProb * 0.07

                        p_act_cnt = self:calcOneStepActPertCounting(p_act_cnt, p_act_ind, _dual_act_change_prob)
                        -- --- Do a test here. Try to not only change feature values in integer. Try directly to use std
                        -- --- This is an interesting try. It seems that it does not generate better result than integer
                        -- --- perturbed action countings. The result is similar. And training is slower.
                        -- p_act_cnt = p_act_cnt * torch.normal(1, 0.22)
                        -- if p_act_cnt > 2 then
                        --     p_act_cnt = 2
                        -- elseif p_act_cnt < -2 then
                        --     p_act_cnt = -2
                        -- end
                        -- perturb action counting
                        input[self.opt.batchSize+i][p_act_ind] = input[self.opt.batchSize+i][p_act_ind] + p_act_cnt
                        -- make sure the counting is non-negative
                        if input[self.opt.batchSize+i][p_act_ind] < 0 then
                            input[self.opt.batchSize+i][p_act_ind] = 0
                        end
                    end
                end
            else
                -- if output_act[i] == self.CIFr.usrActInd_end
                -- then purterb all other action features according to action frequency (not considering action time)
                local actPtbCntForEnd = {1.0, 0.85, 0.3}  -- This is the probability under which the action counting will be perturbed
                for k=1, #actPtbCntForEnd do
                    -- chance of not perturbing
                    if torch.uniform() > actPtbCntForEnd[k] then
                        break
                    end
                    -- otherwise, perturb action values
                    local endPtrSeed = torch.uniform()
                    for pai=1, self.actFreqSortResCum:size()[1] do
                        if endPtrSeed <= self.actFreqSortResCum[pai] then
                            local p_act_ind = self.actFreqSortRank[pai]
                            local p_act_cnt = 1
                            -- When perturb features for synthetic data points at the end of a sequence, we set larger
                            -- probability to see an increase of total action counting to reflect that the ending of
                            -- the game is activated at a certain amount of time
                            if torch.uniform() > 0.65 then
                                p_act_cnt = -1
                            end
                            -- If std is large, then make the case possible to change act count larger than 2
                            -- if math.abs(self.actFreqTotal[p_act_ind]) > 2 and torch.uniform() < 0.3 then
                            --     p_act_cnt = p_act_cnt * 2
                            -- end
                            p_act_cnt = self:calcOneStepActPertCounting(p_act_cnt, p_act_ind, 0.45)
                            -- --- Do a test here. Try to not only change feature values in integer. Try directly to use std
                            -- --- This is an interesting try. It seems that it does not generate better result than integer
                            -- --- perturbed action countings. The result is similar. And training is slower.
                            -- p_act_cnt = p_act_cnt * torch.normal(1, 0.22)
                            -- if p_act_cnt > 2 then
                            --     p_act_cnt = 2
                            -- elseif p_act_cnt < -2 then
                            --     p_act_cnt = -2
                            -- end
                            -- perturb action counting
                            input[self.opt.batchSize+i][p_act_ind] = input[self.opt.batchSize+i][p_act_ind] + p_act_cnt
                            -- make sure the counting is non-negative
                            if input[self.opt.batchSize+i][p_act_ind] < 0 then
                                input[self.opt.batchSize+i][p_act_ind] = 0
                            end

                            break
                        end
                    end
                end
            end

        end

    end

end

--- Add random noise to input feature set
function CIUserSimulator:UserSimDataAddRandNoise(input, isRNNForm, rndNoiseStd)
    -- We add random normal noise to the input feature values
    if isRNNForm then
        for j = 1, self.opt.lstmHist do
            input[j]:add(torch.randn(input[j]:size()) * rndNoiseStd)
        end
    else
        input:add(torch.randn(input:size()) * rndNoiseStd)
    end
end

return CIUserSimulator
