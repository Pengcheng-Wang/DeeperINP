--
-- User: pwang8
-- Date: 1/23/17
-- Time: 12:43 PM
-- Implement a classification problem to predict user's next action in CI
-- user simulation model. Definition of user's state features and actions
-- are in the CI ijcai document.
--

require 'torch'
require 'nn'
require 'nnx'   -- I suspect if this lib is still needed, on Jul 19, 2017
require 'optim'
require 'rnn'
local nninit = require 'nninit'
local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'

local CIUserActsPredictor = classic.class('UserActsPredictor')

function CIUserActsPredictor:_init(CIUserSimulator, opt)

    -- batch size?
    if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
        error('LBFGS should not be used with small mini-batches; 1000 is recommended')
    end

    self.ciUserSimulator = CIUserSimulator
    self.opt = opt
    ----------------------------------------------------------------------
    -- define model to train
    -- on the 15-class classification problem
    --
    classes = {}
    for i=1, CIUserSimulator.CIFr.usrActInd_end do classes[i] = i end
    self.inputFeatureNum = CIUserSimulator.realUserDataStates[1]:size()[1]

    if opt.ciunet == '' then
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
                expert:add(nn.Reshape(self.inputFeatureNum))
                expert:add(nn.Linear(self.inputFeatureNum, 32))
                expert:add(nn.ReLU())
                if opt.dropoutUSim > 0 then expert:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any
                expert:add(nn.Linear(32, 24))
                expert:add(nn.ReLU())
                if opt.dropoutUSim > 0 then expert:add(nn.Dropout(opt.dropoutUSim)) end -- apply dropout, if any
                expert:add(nn.Linear(24, #classes))
                expert:add(nn.LogSoftMax())
                experts:add(expert)
            end

            gater = nn.Sequential()
            gater:add(nn.Reshape(self.inputFeatureNum))
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
            self.model:add(nn.Linear(24, #classes))
            self.model:add(nn.LogSoftMax())
            ------------------------------------------------------------

        elseif opt.uppModel == 'linear' then
            ------------------------------------------------------------
            -- simple linear model: logistic regression
            ------------------------------------------------------------
            self.model:add(nn.Reshape(self.inputFeatureNum))
            self.model:add(nn.Linear(self.inputFeatureNum, #classes))
            self.model:add(nn.LogSoftMax())
            ------------------------------------------------------------

        elseif opt.uppModel == 'lstm' then
            ------------------------------------------------------------
            -- lstm
            ------------------------------------------------------------
            self.model:add(nn.Reshape(self.inputFeatureNum))
            nn.FastLSTM.bn = true
            local lstm
            if opt.uSimGru == 0 then
                lstm = nn.FastLSTM(self.inputFeatureNum, opt.lstmHd, opt.uSimLstmBackLen, nil, nil, nil, opt.dropoutUSim) -- the 3rd param, [rho], the maximum amount of backpropagation steps to take back in time, default value is 9999
                TableSet.fastLSTMForgetGateInit(lstm, opt.dropoutUSim, opt.lstmHd, nninit)
            else
                lstm = nn.GRU(self.inputFeatureNum, opt.lstmHd, opt.uSimLstmBackLen, opt.dropoutUSim)   -- GRU implements its RNN dropout, but does not have built-in batch normalization, as it is for FastLSTM
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
                else
                    lstmL2 = nn.GRU(opt.lstmHd, opt.lstmHdL2, opt.uSimLstmBackLen, opt.dropoutUSim)
                end
                lstmL2:remember('both')
                self.model:add(lstmL2)
                self.model:add(nn.NormStabilizer())
            end
            if opt.lstmHdL2 == 0 then
                self.model:add(nn.Linear(opt.lstmHd, #classes))
            else
                self.model:add(nn.Linear(opt.lstmHdL2, #classes))
            end

            self.model:add(nn.LogSoftMax())
            self.model = nn.Sequencer(self.model)
            ------------------------------------------------------------

        else
            print('Unknown model type')
            cmd:text()
            error()
        end

        -- params init
        local uapLinearLayers = self.model:findModules('nn.Linear')
        for l = 1, #uapLinearLayers do
            uapLinearLayers[l]:init('weight', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)}):init('bias', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)})
        end
    elseif opt.ciunet == 'rlLoad' then  -- If need reload a trained uap model in the RL training/evaluation, not for training uap anymore
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
    self.uapCriterion = nn.ClassNLLCriterion()
    if opt.uppModel == 'lstm' then
        self.uapCriterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
    end

    self.trainEpoch = 1
    -- this matrix records the current confusion across classes
    self.uapConfusion = optim.ConfusionMatrix(classes)

    -- log results to files
    self.uapTrainLogger = optim.Logger(paths.concat('userModelTrained', opt.save, 'train.log'))
    self.uapTestLogger = optim.Logger(paths.concat('userModelTrained', opt.save, 'test.log'))
    self.uapTestLogger:setNames{'Epoch', 'Act Test acc.'}

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
                    for i=1, padi do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i] = torch.Tensor(self.ciUserSimulator.userStateFeatureCnt):fill(0)
                        self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i] = self.ciUserSimulator.realUserDataActs[indSeqHead]  -- duplicate the 1st user action for padded states
                    end
                    for i=1, opt.lstmHist-padi do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i+padi] = self.ciUserSimulator.realUserDataStates[indSeqHead+i-1]
                        self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i+padi] = self.ciUserSimulator.realUserDataActs[indSeqHead+i-1]
                    end
                    if padi == opt.lstmHist-1 then
                        self.rnnRealUserDataStarts[#self.rnnRealUserDataStarts+1] = #self.rnnRealUserDataStates     -- This is the start of a user's record
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
                    for i=1, opt.lstmHist do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i] = self.ciUserSimulator.realUserDataStates[indSeqHead+i-1]
                        self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i] = self.ciUserSimulator.realUserDataActs[indSeqHead+i-1]
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
    self.rnnRealUserDataStartsTest = {}
    self.rnnRealUserDataEndsTest = {}
    self.rnnRealUserDataPadTest = torch.Tensor(#self.ciUserSimulator.realUserDataStartLinesTest):fill(0)    -- indicating whether data has padding at head (should be padded)
    if self.opt.ciuTType == 'train' or self.opt.ciuTType == 'train_tr' then
        if opt.uppModel == 'lstm' then
            local indSeqHeadTest = 1
            local indSeqTailTest = opt.lstmHist
            local indUserSeqTest = 1    -- user id ptr. Use this to get the tail of each trajectory
            while indSeqTailTest <= #self.ciUserSimulator.realUserDataStatesTest do
                if self.rnnRealUserDataPadTest[indUserSeqTest] < 1 then
                    for padi = opt.lstmHist-1, 1, -1 do
                        self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest + 1] = {}
                        self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest + 1] = {}
                        for i=1, padi do
                            self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest][i] = torch.Tensor(self.ciUserSimulator.userStateFeatureCnt):fill(0)
                            self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest][i] = self.ciUserSimulator.realUserDataActsTest[indSeqHeadTest]  -- duplicate the 1st user action for padded states
                        end
                        for i=1, opt.lstmHist-padi do
                            self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest][i+padi] = self.ciUserSimulator.realUserDataStatesTest[indSeqHeadTest+i-1]
                            self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest][i+padi] = self.ciUserSimulator.realUserDataActsTest[indSeqHeadTest+i-1]
                        end
                        if padi == opt.lstmHist-1 then
                            self.rnnRealUserDataStartsTest[#self.rnnRealUserDataStartsTest+1] = #self.rnnRealUserDataStatesTest     -- This is the start of a user's record
                        end
                        if indSeqHeadTest+(opt.lstmHist-padi)-1 == self.ciUserSimulator.realUserDataEndLinesTest[indUserSeqTest] then
                            self.rnnRealUserDataPadTest[indUserSeqTest] = 1
                            break   -- if padding tail is going to outrange this user record's tail, break
                        end
                    end
                    self.rnnRealUserDataPadTest[indUserSeqTest] = 1
                else
                    if indSeqTailTest <= self.ciUserSimulator.realUserDataEndLinesTest[indUserSeqTest] then
                        self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest + 1] = {}
                        self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest + 1] = {}
                        for i=1, opt.lstmHist do
                            self.rnnRealUserDataStatesTest[#self.rnnRealUserDataStatesTest][i] = self.ciUserSimulator.realUserDataStatesTest[indSeqHeadTest+i-1]
                            self.rnnRealUserDataActsTest[#self.rnnRealUserDataActsTest][i] = self.ciUserSimulator.realUserDataActsTest[indSeqHeadTest+i-1]
                        end
                        indSeqHeadTest = indSeqHeadTest + 1
                        indSeqTailTest = indSeqTailTest + 1
                    else
                        self.rnnRealUserDataEndsTest[#self.rnnRealUserDataEndsTest+1] = #self.rnnRealUserDataStatesTest     -- This is the end of a user's record
                        indUserSeqTest = indUserSeqTest + 1 -- next user's records
                        indSeqHeadTest = self.ciUserSimulator.realUserDataStartLinesTest[indUserSeqTest]
                        indSeqTailTest = indSeqHeadTest + opt.lstmHist - 1
                    end
                end
            end
            self.rnnRealUserDataEndsTest[#self.rnnRealUserDataEndsTest+1] = #self.rnnRealUserDataStatesTest     -- Set the end of the last user's record
            -- There are in total 15509 sequences if histLen is 3. 14707 if histLen is 5. 15108 if histLen is 4. 15911 if histLen is 2.
        end
    end


    -- retrieve parameters and gradients
    -- have to put these lines here below the gpu setting
    self.uapParam, self.uapDParam = self.model:getParameters()


    ----- :testing the Pearson's correlation calc function
    --for i=1, self.ciUserSimulator.realUserDataStates[1]:size()[1] do
    --    for j=i+1, self.ciUserSimulator.realUserDataStates[1]:size()[1] do
    --        local pcv = self.ciUserSimulator:PearsonCorrelation(i, j)
    --        print('Pearson\'s correlation between feature',i, 'and',j,'is:',pcv)
    --    end
    --end
    ----- testing the prior action appearance freq calculation
    --print(self.ciUserSimulator.actRankPriorStep)
    -- print(self.ciUserSimulator.featStdDev)
    --print(self.ciUserSimulator:PearsonCorrelation(1,2), self.ciUserSimulator:PearsonCorrelation(1,1))

end


-- training function
function CIUserActsPredictor:trainOneEpoch()
    -- local vars
    local time = sys.clock()

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. self.trainEpoch .. ' [batchSize = ' .. self.opt.batchSize .. ']')
    local inputs
    local targets
    local t = 1
    local lstmIter = 1  -- lstm iterate for each squence starts from this value
    local epochDone = false
    while not epochDone do
        if self.opt.uppModel ~= 'lstm' then
            -- create mini batch
            inputs = torch.Tensor(self.opt.batchSize, self.inputFeatureNum)
            targets = torch.Tensor(self.opt.batchSize)
            local k = 1
            for i = t, math.min(t+self.opt.batchSize-1, #self.ciUserSimulator.realUserDataStates) do
                -- load new sample
                local input = self.ciUserSimulator.realUserDataStates[i]    -- :clone() -- if preprocess is called, clone is not needed, I believe
                -- need do preprocess for input features
                input = self.ciUserSimulator:preprocessUserStateData(input, self.opt.prepro)
                local target = self.ciUserSimulator.realUserDataActs[i]
                inputs[k] = input
                targets[k] = target
                k = k + 1
            end

            -- at the end of dataset, if it could not be divided into full batch
            if k ~= self.opt.batchSize + 1 then
                while k <= self.opt.batchSize do
                    local randInd = torch.random(1, #self.ciUserSimulator.realUserDataStates)
                    inputs[k] = self.ciUserSimulator:preprocessUserStateData(self.ciUserSimulator.realUserDataStates[randInd], self.opt.prepro)
                    targets[k] = self.ciUserSimulator.realUserDataActs[randInd]
                    k = k + 1
                end
            end

            t = t + self.opt.batchSize
            if t > #self.ciUserSimulator.realUserDataStates then
                epochDone = true
            end

            self.ciUserSimulator:UserSimDataAugment()
            if self.opt.gpu > 0 then
                inputs = inputs:cuda()
                targets = targets:cuda()
            end

        else
            -- lstm
            inputs = {}
            targets = {}
            local k
            for j = 1, self.opt.lstmHist do
                inputs[j] = torch.Tensor(self.opt.batchSize, self.inputFeatureNum)
                targets[j] = torch.Tensor(self.opt.batchSize)
                k = 1
                for i = lstmIter, math.min(lstmIter+self.opt.batchSize-1, #self.rnnRealUserDataStates) do
                    local input = self.rnnRealUserDataStates[i][j]
                    input = self.ciUserSimulator:preprocessUserStateData(input, self.opt.prepro)
                    local target = self.rnnRealUserDataActs[i][j]
                    inputs[j][k] = input
                    targets[j][k] = target
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
                        local target = self.rnnRealUserDataActs[randInd][j]
                        inputs[j][k] = input
                        targets[j][k] = target
                    end
                    k = k + 1
                end
            end

            lstmIter = lstmIter + self.opt.batchSize
            if lstmIter > #self.rnnRealUserDataStates then
                epochDone = true
            end

            if self.opt.gpu > 0 then
                nn.utils.recursiveType(inputs, 'torch.CudaTensor')
                nn.utils.recursiveType(targets, 'torch.CudaTensor')
            end

        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- just in case:
            collectgarbage()

            -- get new parameters
            if x ~= self.uapParam then
                self.uapParam:copy(x)
            end

            -- reset gradients
            self.uapDParam:zero()

            -- evaluate function for complete mini batch
            local outputs = self.model:forward(inputs)
            local f = self.uapCriterion:forward(outputs, targets)

            -- estimate df/dW
            local df_do = self.uapCriterion:backward(outputs, targets)
            self.model:backward(inputs, df_do)

            -- penalties (L1 and L2):
            if self.opt.coefL1 ~= 0 or self.opt.coefL2 ~= 0 then
                -- locals:
                local norm,sign= torch.norm,torch.sign

                -- Loss:
                f = f + self.opt.coefL1 * norm(self.uapParam,1)
                f = f + self.opt.coefL2 * norm(self.uapParam,2)^2/2

                -- Gradients:
                self.uapDParam:add( sign(self.uapParam):mul(self.opt.coefL1) + self.uapParam:clone():mul(self.opt.coefL2) )
            end

            -- update self.uapConfusion
            if self.opt.uppModel == 'lstm' then
                for j = 1, self.opt.lstmHist do
                    for i = 1,self.opt.batchSize do
                        self.uapConfusion:add(outputs[j][i], targets[j][i])
                    end
                end
            else
                for i = 1,self.opt.batchSize do
                    self.uapConfusion:add(outputs[i], targets[i])
                end
            end

            -- return f and df/dX
            return f, self.uapDParam
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
            optim.lbfgs(feval, self.uapParam, lbfgsState)

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
            optim.sgd(feval, self.uapParam, sgdState)

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
            optim.adam(feval, self.uapParam, adamState)

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
            optim.rmsprop(feval, self.uapParam, rmspropState)

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

    -- print self.uapConfusion matrix
    --    print(self.uapConfusion)
    self.uapConfusion:updateValids()
    local confMtxStr = 'average row correct: ' .. (self.uapConfusion.averageValid*100) .. '% \n' ..
    'average rowUcol correct (VOC measure): ' .. (self.uapConfusion.averageUnionValid*100) .. '% \n' ..
    ' + global correct: ' .. (self.uapConfusion.totalValid*100) .. '%'
    print(confMtxStr)
    self.uapTrainLogger:add{['% mean class accuracy (train set)'] = self.uapConfusion.totalValid * 100}


    -- save/log current net
    local filename = paths.concat('userModelTrained', self.opt.save, 'uap.t7')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    --    if paths.filep(filename) then
    --        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    --    end
    --    print('<trainer> saving ciunet to '..filename)
    --    torch.save(filename, self.model)

    if self.trainEpoch % 10 == 0 and self.opt.ciuTType == 'train' then
        filename = paths.concat('userModelTrained', self.opt.save, string.format('%d', self.trainEpoch)..'_'..string.format('%.2f', self.uapConfusion.totalValid*100)..'uap.t7')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('<trainer> saving periodly trained ciunet to '..filename)
        torch.save(filename, self.model)
    end

    if (self.opt.ciuTType == 'train' or self.opt.ciuTType == 'train_tr') and self.trainEpoch % self.opt.testOnTestFreq == 0 then
        local testAccu = self:testActPredOnTestDetOneEpoch()
        print('<Act prediction accuracy at epoch '..string.format('%d', self.trainEpoch)..' on test set > '..string.format('%.2f%%', testAccu*100))
        self.uapTestLogger:add{string.format('%d', self.trainEpoch), string.format('%.5f%%', testAccu*100)}
    end

    self.uapConfusion:zero()
    -- next epoch
    self.trainEpoch = self.trainEpoch + 1
end

-- evaluation function on test/train_validation set
function CIUserActsPredictor:testActPredOnTestDetOneEpoch()
    -- Confusion matrix for action prediction (15 class)
    --    local actPredTP = torch.Tensor(self.ciUserSimulator.CIFr.usrActInd_end):fill(1e-3)
    --    local actPredFP = torch.Tensor(self.ciUserSimulator.CIFr.usrActInd_end):fill(1e-3)
    --    local actPredFN = torch.Tensor(self.ciUserSimulator.CIFr.usrActInd_end):fill(1e-3)

    if self.opt.uppModel == 'lstm' then
        -- uSimShLayer == 0 and lstm model
        self.model:evaluate()
        self.model:forget()

        local tabState = {}
        for j=1, self.opt.lstmHist do
            local prepUserState = torch.Tensor(#self.rnnRealUserDataStatesTest, self.ciUserSimulator.userStateFeatureCnt)
            for k=1, #self.rnnRealUserDataStatesTest do
                prepUserState[k] = self.ciUserSimulator:preprocessUserStateData(self.rnnRealUserDataStatesTest[k][j], self.opt.prepro)
            end
            tabState[j] = prepUserState
        end
        if self.opt.gpu > 0 then
            nn.utils.recursiveType(tabState, 'torch.CudaTensor')
        end
        local nll_acts = self.model:forward(tabState)

        self.uapConfusion:zero()
        nn.utils.recursiveType(nll_acts, 'torch.FloatTensor')
        for i=1, #self.rnnRealUserDataStatesTest do
            self.uapConfusion:add(nll_acts[self.opt.lstmHist][i], self.rnnRealUserDataActsTest[i][self.opt.lstmHist])
        end
        self.uapConfusion:updateValids()
        local tvalid = self.uapConfusion.totalValid
        self.uapConfusion:zero()
        return tvalid

    else
        -- uSimShLayer == 0 and not lstm models
        self.model:evaluate()

        local prepUserState = torch.Tensor(#self.ciUserSimulator.realUserDataStatesTest, self.ciUserSimulator.userStateFeatureCnt)
        for i=1, #self.ciUserSimulator.realUserDataStatesTest do
            prepUserState[i] = self.ciUserSimulator:preprocessUserStateData(self.ciUserSimulator.realUserDataStatesTest[i], self.opt.prepro)
        end
        if self.opt.gpu > 0 then
            prepUserState = prepUserState:cuda()
        end
        local nll_acts = self.model:forward(prepUserState)

        self.uapConfusion:zero()
        nll_acts:float()     -- set nll_rewards back to cpu mode (in main memory)
        for i=1, #self.ciUserSimulator.realUserDataStatesTest do
            self.uapConfusion:add(nll_acts[i], self.ciUserSimulator.realUserDataActsTest[i])
        end
        self.uapConfusion:updateValids()
        local tvalid = self.uapConfusion.totalValid
        self.uapConfusion:zero()
        return tvalid
    end

end

return CIUserActsPredictor
