--
-- User: pwang8
-- Date: 1/15/17
-- Time: 4:35 PM
-- This file is created to read trace and survey files of the CI narrative adaptation experiments.
-- The original purpose of this file is to create a lua version of file reader as the original
-- python file reader used in previous experiments. (The whole thing has to be done!)
--

local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'

local CIFileReader = classic.class('CIFileReader')

local posterReadMinTime = 2.0
local bookReadMinTime = 5.0

local invalid_set = {}
--TableSet.addToSet(invalid_set, '100-0025')
--TableSet.addToSet(invalid_set, '100-0026')
local invalid_cnt = 0

-- Creates CI8 trace and survey file readers
function CIFileReader:_init(opt)
    self.usrActInd_posterRead = 1
    self.usrActInd_bookRead = 2
    self.usrActInd_talkKim = 3
    self.usrActInd_askTeresaSymp = 4
    self.usrActInd_askBryceSymp = 5
    self.usrActInd_talkQuentin = 6
    self.usrActInd_talkRobert = 7
    self.usrActInd_talkFord = 8
    self.usrActInd_quiz = 9
    self.usrActInd_testObject = 10
    self.usrActInd_submitWorksheet = 11
    self.usrActInd_BryceRevealActOne = 12
    self.usrActInd_QuentinRevealActOne = 13
    self.usrActInd_KimLetQuentinRevealActOne = 14
    self.usrActInd_end = 15

    self.traceFilePath = 'data/training-log-corpus.log'  --'data/training-survey-corpus.csv'
    self.userStateGamePlayFeatureCnt = 18
    self.usrStateFeatureInd_TeresaSymp = 15
    self.usrStateFeatureInd_BryceSymp = 16
    self.usrStateFeatureInd_WorksheetLevel = 17
    self.usrStateFeatureInd_PresentQuiz = 18

    -- Adaptation type index
    self.ciAdp_TeresaSymp = 1 -- 3 acts. (1-3) act-1: max detail, act-2: moderate detail, act-3: minimal detail. (act1--1.0, act3--0.33). So y=(4-x)/3
    self.ciAdp_BryceSymp = 2 -- 2 acts. (4-5) act-1: moderate detail, act-2: minimal detail. (act1--1.0, act2--0.5). So y=(3-x)/2
    self.ciAdp_WorksheetLevel = 3 -- 3 acts. (6-8) act-1: minimal detail, act-2: moderate detail, act-3: maximal detail. (act1-0.33, act3-1). y=x/3
    self.ciAdp_PresentQuiz = 4 -- 2 acts. (9-10) act-1: quiz, act-2: no-quiz. (act1-quiz-1.0, act2-no_quiz-0). y=2-x

    self.ciAdpActRange_TeresaSymp = {1,3} -- 3 acts. (1-3) act-1: max detail, act-2: moderate detail, act-3: minimal detail. (act1--1.0, act3--0.33). So y=(4-x)/3
    self.ciAdpActRange_BryceSymp = {4,5} -- 2 acts. (4-5) act-1: moderate detail, act-2: minimal detail. (act1--1.0, act2--0.5). So y=(3-x)/2
    self.ciAdpActRange_WorksheetLevel = {6,8} -- 3 acts. (6-8) act-1: minimal detail, act-2: moderate detail, act-3: maximal detail. (act1-0.33, act3-1). y=x/3
    self.ciAdpActRange_PresentQuiz = {9,10} -- 2 acts. (9-10) act-1: quiz, act-2: no-quiz. (act1-quiz-1.0, act2-no_quiz-0). y=2-x
    self.ciAdpActRanges = {self.ciAdpActRange_TeresaSymp, self.ciAdpActRange_BryceSymp, self.ciAdpActRange_WorksheetLevel, self.ciAdpActRange_PresentQuiz}

    -- Read data from CSV to tensor
    local traceFile = io.open(self.traceFilePath, 'r')
--    local header = traceFile:read()

    self.traceData = {}
    self.AdpTeresaSymptomAct = {}
    self.AdpBryceSymptomAct = {}
    self.AdpPresentQuizAct = {}
    self.AdpWorksheetLevelAct = {}
    self.KimTriggerQuentinReveal = {}
    self.talkCntQuentin = {}
    self.talkCntRobert = {}
    self.talkCntFord = {}

    local curId = ''
    local searchNextAdpPresentQuiz = false
    local talkRobFordQuentinLine = 0
    local talkCntRobFordQuentin
    local searchQuizConfirm = false
    local quizEarnMoreTestsinLine = 0
    local id_cnt = 0
    local tmp_inv_set = {}

    local i = 0     -- line number in trace file
    for line in traceFile:lines('*l') do
        i = i + 1
        local oneLine = line:split('|')
        if TableSet.setContains(invalid_set, oneLine[1]) then
            print('Invalid id', oneLine[1], 'line', i)
            os.exit()
        elseif curId ~= oneLine[1] then -- new ID observed
            if curId ~= '' then
--                print('### End action', i)
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_end   -- End action
            end
            id_cnt = id_cnt + 1
            if searchNextAdpPresentQuiz or searchQuizConfirm then
                print('Error in trace file, line', i,
                    '. searchNextAdpPresentQuiz or searchQuizConfirm is true at starting line')
                os.exit()
            end
            curId = oneLine[1]
--            print('@ New stu', curId)
            self.traceData[curId] = {}  -- Assume the first line in each user's trace does not have necessary info
            self.AdpTeresaSymptomAct[curId] = {}
            self.AdpBryceSymptomAct[curId] = {}
            self.AdpPresentQuizAct[curId] = {}
            self.AdpWorksheetLevelAct[curId] = {}

            self.KimTriggerQuentinReveal[curId] = 0 -- This table stores number of events that select-kim-reveal act-3 is chosen
                                                    -- If it is before 1st talk with Quentin,
                                                    -- select-present-quiz will not be triggered by Quentin 1st talk.
            self.talkCntQuentin[curId] = 0
            self.talkCntRobert[curId] = 0
            self.talkCntFord[curId] = 0
        else
            if searchNextAdpPresentQuiz then
                if i > talkRobFordQuentinLine + 3 then
                    print('Error in trace file, line', i, '. select-present-quiz should be in 3 rows from the triggering talk')
                    invalid_cnt = invalid_cnt+1
                    tmp_inv_set[#tmp_inv_set + 1] = curId
                    searchNextAdpPresentQuiz = false  -- this ignore the current parsing error. Otherwise, comment this line
                            -- and uncomment next line, so the program will exit at the wrongly formatted line
                    -- os.exit()
                elseif oneLine[2] == 'DIALOG' and oneLine[6] and (string.sub(oneLine[6], 1, 7) == 'Kimwant' or
                        string.sub(oneLine[6], 1, 7) == 'Kim,the') then     -- player has to talk to Kim first
--                    print('### Delete the talk log on line', talkRobFordQuentinLine)
                    self.traceData[curId][#self.traceData[curId]] = nil     -- delete the most recent record
                    talkCntRobFordQuentin[curId] = talkCntRobFordQuentin[curId] - 1     -- decrease count
                    searchNextAdpPresentQuiz = false
                elseif oneLine[2] == 'ADAPTATION' and oneLine[4] == 'select-present-quiz' then
                    self.AdpPresentQuizAct[curId][#self.traceData[curId]] = tonumber(string.sub(oneLine[6], -1, -1))
                    searchNextAdpPresentQuiz = false
                end
            elseif searchQuizConfirm then   -- earn-more-tests quiz could be taken or retaken. This is detection of 1st quiz
                if (i == quizEarnMoreTestsinLine + 1) and oneLine[8] == 'press-yes' then
--                    print('### Quiz', i)
                    self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_quiz   -- quiz action index
                end
                searchQuizConfirm = false
            elseif oneLine[2] == 'LOOKEND' and oneLine[4]:split('-')[1] == 'poster' and tonumber(oneLine[5]:split('-')[2]) > posterReadMinTime then
--                print('#@# poster', i, tonumber(oneLine[5]:split('-')[2]))
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_posterRead   -- poster reading action index
            elseif oneLine[2] == 'BOOKREAD' and tonumber(oneLine[7]) > bookReadMinTime then
--                print('@!@ book', i, tonumber(oneLine[7]))
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_bookRead   -- book reading action index
            elseif oneLine[2] == 'DIALOG' and oneLine[5] == 'kim' and string.sub(oneLine[6], 1, string.len('Pathogen')) == 'Pathogen' then
--                print('@@@ talk kim', i)
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_talkKim   -- talk with (ask about pathogen) Kim
            elseif oneLine[2] == 'ADAPTATION' and oneLine[4] == 'select-teresa-symptoms-level' then
--                print('### ask Teresa symptom', i)
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_askTeresaSymp   -- Ask Teresa about her symptoms
                self.AdpTeresaSymptomAct[curId][#self.traceData[curId]] = tonumber(string.sub(oneLine[6], -1, -1))
            elseif oneLine[2] == 'ADAPTATION' and oneLine[4] == 'select-bryce-symptoms-level' then
--                print('### ask Bryce symptoms', i)
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_askBryceSymp   -- Ask Bryce about his symptoms
                self.AdpBryceSymptomAct[curId][#self.traceData[curId]] = tonumber(string.sub(oneLine[6], -1, -1))
            elseif oneLine[2] == 'TALK' and oneLine[5] == 'cur-action-talk-quentin' then
--                print('### talk to quentin', i)
                if self.talkCntQuentin[curId] == 0 and self.KimTriggerQuentinReveal[curId] == 0 then -- select-prent-quiz will be triggered when talking with Quentin for 1st time
--                    print('###### first talk with quentin') -- but after talking with Kim
                    searchNextAdpPresentQuiz = true -- and Kim does NOT trigger the Quentin-reveal-action-3 event yet. If select-kim-reveal act-3 is executed, then select-present-quiz will never be triggered from Quentin
                end
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_talkQuentin   -- Talk with Quentin
                self.talkCntQuentin[curId] = self.talkCntQuentin[curId] + 1
                talkCntRobFordQuentin = self.talkCntQuentin
                talkRobFordQuentinLine = i
            elseif oneLine[2] == 'TALK' and oneLine[5] == 'cur-action-talk-robert' then
--                print('### talk to robert', i)
                if self.talkCntRobert[curId] == 0 then -- select-prent-quiz will be triggered when talking with Robert for 1st time
--                    print('###### first talk with robert')
                    searchNextAdpPresentQuiz = true
                end
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_talkRobert   -- Talk with Robert
                self.talkCntRobert[curId] = self.talkCntRobert[curId] + 1
                talkCntRobFordQuentin = self.talkCntRobert
                talkRobFordQuentinLine = i
            elseif oneLine[2] == 'TALK' and oneLine[5] == 'cur-action-talk-ford' then
--                print('### talk to ford', i)
                if self.talkCntFord[curId] == 0 then -- select-prent-quiz will be triggered when talking with Ford for 1st time
--                    print('###### first talk with ford')
                    searchNextAdpPresentQuiz = true
                end
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_talkFord   -- Talk with Robert
                self.talkCntFord[curId] = self.talkCntFord[curId] + 1
                talkCntRobFordQuentin = self.talkCntFord
                talkRobFordQuentinLine = i
            elseif oneLine[2] == 'PDAOPEN' and oneLine[4] == 'earn-more-tests' then     -- detect of initial quiz
                searchQuizConfirm = true
                quizEarnMoreTestsinLine = i
            elseif oneLine[2] == 'PDAUSE' and oneLine[8] == 'press-retakequiz' then     -- detect of retaken quiz
--                print('### Quiz-re', i)
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_quiz   -- quiz action index
            elseif oneLine[2] == 'TESTOBJECT' and oneLine[5] ~= 'noenergy' and
                    oneLine[4] ~= 'NoObject' and oneLine[4] ~= 'MultipleObjects' then
--                print('### Test-obj', i)
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_testObject   -- test object action index
            elseif oneLine[2] == 'ADAPTATION' and oneLine[4] == 'select-worksheet-level' then
--                print('### Submit wrong worksheet', i)
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_submitWorksheet   -- submit worksheet wrong
                self.AdpWorksheetLevelAct[curId][#self.traceData[curId]] = tonumber(string.sub(oneLine[6], -1, -1))
            elseif oneLine[2] == 'DIALOG' and oneLine[5] == 'bryce' and string.sub(oneLine[6], 1, 11) == 'BeforeIgots' then
--                print('### Bryce reveal info', i)
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_BryceRevealActOne   -- Bryce reveals info
            elseif oneLine[2] == 'DIALOG' and oneLine[5] == 'quentin' and string.sub(oneLine[6], 1, 11) == 'Thereissome' then
--                print('### Quentin reveal info', i)
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_QuentinRevealActOne   -- Quentin reveals info
            elseif oneLine[2] == 'ADAPTATION' and oneLine[4] == 'select-kim-reveal' and
                    oneLine[6] == 'selected-3' then
--                print('### Kim let Quentin reveal', i)
                self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_KimLetQuentinRevealActOne   -- Kim let Quentin reveal
                self.KimTriggerQuentinReveal[curId] = 1
            end
        end
--        if i == 3000 then
--            break
--        end
    end
--    print('### End action', i)
    self.traceData[curId][#self.traceData[curId] + 1] = self.usrActInd_end   -- End action for the last user

    print('Trace invalid', invalid_cnt)
    print('Trace records number:', id_cnt)
--    print('!!! inv', tmp_inv_set)
--    print('@@@', self.traceData)
--    print('### Teresa adp', self.AdpTeresaSymptomAct)
--    print('### Bryce adp', self.AdpBryceSymptomAct)
--    print('### PresentQ adp', self.AdpPresentQuizAct)
--    print('### SubSheet adp', self.AdpWorksheetLevelAct)

    traceFile:close()

    --- Read from survey file
    self.surveyFilePath = 'data/training-survey-corpus.csv'
    -- Read data from CSV to tensor
    self.userStateSurveyFeatureCnt = 3  -- gender, game freqency, pre-score
    local surveyFile = io.open(self.surveyFilePath, 'r')
    surveyFile:read()  -- The 1st line contains column names
    self.surveyData = {}

    i = 1   -- line number indicator. The 1st line in survey file contains column names
    for line in surveyFile:lines('*l') do
        i = i + 1
        local oneLine = line:split(',')
        self.surveyData[oneLine[1]] = {}
        self.surveyData[oneLine[1]][1] = tonumber(oneLine[3]) - 1   -- gender
        self.surveyData[oneLine[1]][2] = tonumber(oneLine[4])/5.0   -- game frequency
        self.surveyData[oneLine[1]][3] = tonumber(oneLine[26])/19.0   -- pre-test score
        self.surveyData[oneLine[1]][4] = tonumber(oneLine[92])   -- nlg
    end
    surveyFile:close()

end

--- Check if all records in trace log file follows the Adaptation triggering rule
function CIFileReader:evaluateTraceFile()
    local traceRcdCnt = 0
    local endActs = 0
    for userId, userRcd in pairs(self.traceData) do
        traceRcdCnt = traceRcdCnt + 1
        local talkCntQuentin = 0
        local talkCntRobert = 0
        local talkCntFord = 0
        local KimTriggerQuentinReveal = 0
        local adpCntTeresaSymp = 0
        local adpCntBryceSymp = 0
        local adpCntPresentQuiz = 0
        local adpCntWorksheetLevel = 0
        for time, act in ipairs(userRcd) do
            if act == self.usrActInd_askTeresaSymp then
                if self.AdpTeresaSymptomAct[userId][time] and
                        self.AdpTeresaSymptomAct[userId][time] >= 1 and
                        self.AdpTeresaSymptomAct[userId][time] <= 3 then
--                    print('# AdpTeresa valid', userId, time, self.AdpTeresaSymptomAct[userId][time])
                    adpCntTeresaSymp = adpCntTeresaSymp + 1
                else
                    print('!!! Error. Teresa symp adp does not align', userId, time)
                    os.exit()
                end
            elseif act == self.usrActInd_askBryceSymp then
                if self.AdpBryceSymptomAct[userId][time] and
                        self.AdpBryceSymptomAct[userId][time] >= 1 and
                        self.AdpBryceSymptomAct[userId][time] <= 2 then
--                    print('# AdpBryce valid', userId, time, self.AdpBryceSymptomAct[userId][time])
                    adpCntBryceSymp = adpCntBryceSymp + 1
                else
                    print('!!! Error. Bryce symp adp does not align', userId, time)
                    os.exit()
                end
            elseif act == self.usrActInd_KimLetQuentinRevealActOne then
                KimTriggerQuentinReveal = 1
            elseif act == self.usrActInd_talkQuentin and KimTriggerQuentinReveal == 0 and
                    talkCntQuentin == 0 then
                if self.AdpPresentQuizAct[userId][time] and
                        self.AdpPresentQuizAct[userId][time] >= 1 and
                        self.AdpPresentQuizAct[userId][time] <= 2 then
--                    print('# AdpPresQuiz(Quentin) valid', userId, time, self.AdpPresentQuizAct[userId][time])
                    adpCntPresentQuiz = adpCntPresentQuiz + 1
                    talkCntQuentin = talkCntQuentin + 1
                else
                    print('!!! Error. Quentin Present quiz adp does not align', userId, time)
                    os.exit()
                end
            elseif act == self.usrActInd_talkRobert and talkCntRobert == 0 then
                if self.AdpPresentQuizAct[userId][time] and
                        self.AdpPresentQuizAct[userId][time] >= 1 and
                        self.AdpPresentQuizAct[userId][time] <= 2 then
--                    print('# AdpPresQuiz(Robert) valid', userId, time, self.AdpPresentQuizAct[userId][time])
                    adpCntPresentQuiz = adpCntPresentQuiz + 1
                    talkCntRobert = talkCntRobert + 1
                else
                    print('!!! Error. Robert Present quiz adp does not align', userId, time)
                    os.exit()
                end
            elseif act == self.usrActInd_talkFord and talkCntFord == 0 then
                if self.AdpPresentQuizAct[userId][time] and
                        self.AdpPresentQuizAct[userId][time] >= 1 and
                        self.AdpPresentQuizAct[userId][time] <= 2 then
--                    print('# AdpPresQuiz(Ford) valid', userId, time, self.AdpPresentQuizAct[userId][time])
                    adpCntPresentQuiz = adpCntPresentQuiz + 1
                    talkCntFord = talkCntFord + 1
                else
                    print('!!! Error. Ford Present quiz adp does not align', userId, time)
                    os.exit()
                end
            elseif act == self.usrActInd_submitWorksheet then
                if self.AdpWorksheetLevelAct[userId][time] and
                        self.AdpWorksheetLevelAct[userId][time] >= 1 and
                        self.AdpWorksheetLevelAct[userId][time] <= 3 then
--                    print('# AdpWorksheet valid', userId, time, self.AdpWorksheetLevelAct[userId][time])
                    adpCntWorksheetLevel = adpCntWorksheetLevel + 1
                else
                    print('!!! Error. Worksheet adp does not align', userId, time)
                    os.exit()
                end
            elseif act == self.usrActInd_end then
                endActs = endActs + 1
            end

        end

        -- Check if the # of adps in adp tables are the same as in trace table
        if adpCntTeresaSymp == TableSet.countsInSet(self.AdpTeresaSymptomAct[userId]) then
--            print('# Teresa Adp count matches', userId)
        else
            print('!!! Error. Teresa Adp count matches', userId)
            os.exit()
        end
        if adpCntBryceSymp == TableSet.countsInSet(self.AdpBryceSymptomAct[userId]) then
--            print('# Bryce Adp count matches', userId)
        else
            print('!!! Error. Bryce Adp count matches', userId)
            os.exit()
        end
        if adpCntPresentQuiz == TableSet.countsInSet(self.AdpPresentQuizAct[userId]) then
--            print('# Present quiz Adp count matches', userId)
        else
            print('!!! Error. Present quiz Adp count matches', userId)
            os.exit()
        end
        if adpCntWorksheetLevel == TableSet.countsInSet(self.AdpWorksheetLevelAct[userId]) then
--            print('# Worksheet Adp count matches', userId)
        else
            print('!!! Error. Worksheet Adp count matches', userId)
            os.exit()
        end

    end

    if traceRcdCnt == endActs then
--        print('# Ends count matches', endActs)
    else
        print('!!! Error. Ends count matches. traceRcdCnt:', traceRcdCnt, 'endActs:', endActs)
        os.exit()
    end

    print('Trace data valid!')

end

--- Check if survey data from survey file matches with trace data, and check if values in
--- each field are in valid scope
function CIFileReader:evaluateSurveyData()
    for userId, _ in pairs(self.traceData) do
        if self.surveyData[userId] == nil then
            print('!!! Error. Survey data is nil for user id:', userId)
            os.exit()
        elseif self.surveyData[userId][1] ~= 0 and self.surveyData[userId][1] ~= 1 then
            print('!!! Error. Gender field in survey data is not valid:', userId)   -- gender field
            os.exit()
        elseif self.surveyData[userId][2] < 0 or self.surveyData[userId][2] > 1 then
            print('!!! Error. Game freqency field in survey data is not valid:', userId)
            os.exit()
        elseif self.surveyData[userId][3] < 0 or self.surveyData[userId][3] > 1 then
            print('!!! Error. Pre-test score field in survey data is not valid:', userId)
            os.exit()
        elseif self.surveyData[userId][4] < -1 or self.surveyData[userId][4] > 1 then
            print('!!! Error. NLG field in survey data is not valid:', userId)
            os.exit()
        end
    end

    -- Check if records in survey and trace files have same count
    if TableSet.countsInSet(self.surveyData) ~= TableSet.countsInSet(self.traceData) then
        print('!!! Error. Survey and trace file record numbers do not match')
        os.exit()
    end

    print('Survey data is valid!')
end

return CIFileReader

