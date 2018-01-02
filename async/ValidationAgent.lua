local _ = require 'moses'
local AsyncModel = require 'async/AsyncModel'
local Evaluator = require 'Evaluator'
local Experience = require 'Experience'
local CircularQueue = require 'structures/CircularQueue'
local classic = require 'classic'
local gnuplot = require 'gnuplot'
require 'classic.torch'
local TableSet = require 'MyMisc.TableSetMisc'

local ValidationAgent = classic.class('ValidationAgent')

local TINY_EPSILON = 1e-6

function ValidationAgent:_init(opt, theta, atomic)
  log.info('creating ValidationAgent')
  local asyncModel = AsyncModel(opt)
  self.env, self.model = asyncModel:getEnvAndModel()
  self.policyNet_ = asyncModel:createNet()

  self.lstm = opt.recurrent and self.policyNet_:findModules('nn.FastLSTM')[1]

  self.theta_ = self.policyNet_:getParameters()
  self.theta = theta

  self.atomic = atomic
  self._id = opt._id

  -- Validation variables
  self.valSize = opt.valSize
  self.losses = {}
  self.avgV = {} -- Running average of V(s')
  self.avgTdErr = {} -- Running average of TD-error δ
  self.valScores = {} -- Validation scores (passed from main script)
  self.normScores = {} -- Normalised validation scores (passed from main script)

  self.m = opt.actionSpec[3][2] - opt.actionSpec[3][1] + 1 -- Number of discrete actions
  self.actionOffset = 1 - opt.actionSpec[3][1] -- Calculate offset if first action is not indexed as 1

  self.env:training()

  self.stateBuffer = CircularQueue(opt.recurrent and 1 or opt.histLen, opt.Tensor, opt.stateSpec[2])
  self.progFreq = opt.progFreq
  self.Tensor = opt.Tensor

  self.reportWeights = opt.reportWeights
  self.valSteps = opt.valSteps
  self.evaluator = Evaluator(opt.game)

  opt.batchSize = opt.valSize -- override in this thread ONLY
  self.valMemory = Experience(opt.valSize + 3, opt, true)

  self.bestValScore = -math.huge

  self.selectAction = self.eGreedyAction
  if opt.actor_critic then self.selectAction = self.probabilisticAction end  -- todo:pwang8. Might need to change this for all actor-critic methods. Dec 29, 2017

  self.opt = opt
  -- Sorry, adding ugly code here again, just for CI data compatability
  self.CIActAdpBound = {{1, 3}, {4, 5}, {6, 8}, {9, 10} }

  classic.strict(self)
end


function ValidationAgent:start()
  log.info('ValidationAgent | filling ValMemory ')
  local reward, terminal = 0, false
  local rawObservation, adpType = self.env:start()   -- Todo: pwang8. This has been changed a little for compatibility with CI sim
  local action = 1
  for i=1,self.valSize+1 do
    local observation = self.model:preprocess(rawObservation)
    self.valMemory:store(reward, observation, terminal, action)
    if not terminal then
      action = torch.random(1,self.m)

      if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness for CI sim compatibility
        local adpT = 0
        if observation[-1][1][-4] > 0.1 then adpT = 1 elseif observation[-1][1][-3] > 0.1 then adpT = 2 elseif observation[-1][1][-2] > 0.1 then adpT = 3 elseif observation[-1][1][-1] > 0.1 then adpT = 4 end
        assert(adpT >=1 and adpT <= 4)
        action = torch.random(self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2])
      end

      reward, rawObservation, terminal, adpType = self.env:step(action - self.actionOffset)
    else
      reward, terminal = 0, false
      rawObservation, adpType = self.env:start()    -- Todo: pwang8. This has been changed a little for compatibility with CI sim
    end
  end
  log.info('ValidationAgent | ValMemory filled')
end


function ValidationAgent:eGreedyAction(state)
  local epsilon = 0.001 -- Taken from tuned DDQN evaluation

  local Q = self.policyNet_:forward(state):squeeze()

  local actDist = {}  -- Todo: pwang8. Check correctness. This is act selection probability dist (could be used in Importance sampling). Only used in evaluation
  -- If it is CI data, pick up actions according to adpType
  local adpT = 0
  if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
    if state[-1][1][-4] > 0.1 then adpT = 1 elseif state[-1][1][-3] > 0.1 then adpT = 2 elseif state[-1][1][-2] > 0.1 then adpT = 3 elseif state[-1][1][-1] > 0.1 then adpT = 4 end
    assert(adpT >=1 and adpT <= 4)

    -- Calculate the action selection distribution
    local temQsum = 0
    for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
      temQsum = temQsum + math.exp(self.opt.actDistT * Q[j])
    end
    for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
      actDist[j] = math.exp(self.opt.actDistT * Q[j]) / temQsum
    end
  end

  if torch.uniform() < epsilon then
    if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
      return torch.random(self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2]), actDist
    else
      return torch.random(1,self.m), actDist
    end
  end

  if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
    local maxAct = self.CIActAdpBound[adpT][1]
    local maxActQValue = Q[maxAct]
    for i=maxAct+1, self.CIActAdpBound[adpT][2] do
      if Q[i] > maxActQValue then
        maxActQValue = Q[i]
        maxAct = i
      end
    end
    return maxAct, actDist
  else
    local _, maxIdx = Q:max(1)
    return maxIdx[1], actDist
  end
end


function ValidationAgent:probabilisticAction(state)
  local __, probability = table.unpack(self.policyNet_:forward(state))

  local actDist = {}  -- Todo: pwang8. Check correctness. This is act selection probability dist (could be used in Importance sampling). Only used in evaluation
  if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
    -- If it is CI data, pick up actions according to adpType
    local adpT = 0
    if state[-1][1][-4] > 0.1 then adpT = 1 elseif state[-1][1][-3] > 0.1 then adpT = 2 elseif state[-1][1][-2] > 0.1 then adpT = 3 elseif state[-1][1][-1] > 0.1 then adpT = 4 end
    assert(adpT >=1 and adpT <= 4)
    local subAdpActRegion = torch.Tensor(self.CIActAdpBound[adpT][2] - self.CIActAdpBound[adpT][1] + 1):fill(0)
    probability:squeeze()
    for i=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
      subAdpActRegion[i-self.CIActAdpBound[adpT][1]+1] = probability[i]
      actDist[i] = probability[i] + TINY_EPSILON
    end
    -- Have to make sure subAdpActRegion does not sum up to 0 (all 0s) before sent to multinomial()
    subAdpActRegion:add(TINY_EPSILON) -- add a small number to this distribution so it will not sum up to 0
    local regAct = torch.multinomial(subAdpActRegion, 1):squeeze()
    if self.opt.ac_greedy then _, regAct = torch.max(subAdpActRegion, 1) regAct = regAct[1] end
--    print('Display act choice dist: ', subAdpActRegion:squeeze())
    return self.CIActAdpBound[adpT][1] + regAct - 1, actDist
  else
    return torch.multinomial(probability, 1):squeeze(), probability:add(TINY_EPSILON):totable()
  end
end


function ValidationAgent:validate()
  self.theta_:copy(self.theta)
  if self.lstm then self.lstm:forget() end

  self.stateBuffer:clear()
  self.env:evaluate()
  self.policyNet_:evaluate()

  local valStepStrFormat = '%0' .. (math.floor(math.log10(self.valSteps)) + 1) .. 'd'
  local valEpisode = 1
  local valEpisodeScore = 0
  local valTotalScore = 0

  local reward, terminal = 0, false
  local observation, adpType = self.env:start()   -- Todo: pwang8. This has been changed a little for compatibility with CI sim

  for valStep = 1, self.valSteps do
    observation = self.model:preprocess(observation)
    if terminal then
      self.stateBuffer:clear()
    else
      self.stateBuffer:push(observation)
    end
    if not terminal then
      local state = self.stateBuffer:readAll()

      local action = self:selectAction(state)

      reward, observation, terminal, adpType = self.env:step(action - self.actionOffset)
      valEpisodeScore = valEpisodeScore + reward
    else
      if self.lstm then self.lstm:forget() end

      -- Print score every 10 episodes
      if valEpisode % 10 == 0 then
        local avgScore = valTotalScore/math.max(valEpisode - 1, 1)
        log.info('[VAL] Steps: ' .. string.format(valStepStrFormat, valStep) .. '/' .. self.valSteps .. ' | Episode ' .. valEpisode
          .. ' | Score: ' .. valEpisodeScore .. ' | TotScore: ' .. valTotalScore .. ' | AvgScore: %.2f', avgScore)
      end

      -- Start a new episode
      valEpisode = valEpisode + 1
      reward, terminal = 0, false
      observation, adpType = self.env:start()    -- Todo: pwang8. This has been changed a little for compatibility with CI sim
      valTotalScore = valTotalScore + valEpisodeScore -- Only add to total score at end of episode
      valEpisodeScore = reward -- Reset episode score
    end
  end

  -- If no episodes completed then use score from incomplete episode
  if valEpisode == 1 then
    valTotalScore = valEpisodeScore
  end

  log.info('Validated @ '.. self.atomic:get())
  log.info('Total Score: ' .. valTotalScore)
  local valAvgScore = valTotalScore/math.max(valEpisode - 1, 1) -- Only average score for completed episodes in general
  log.info('Average Score: ' .. valAvgScore)
  self.valScores[#self.valScores + 1] = valAvgScore
  local normScore = self.evaluator:normaliseScore(valAvgScore)
  if normScore then
    log.info('Normalised Score: ' .. normScore)
    self.normScores[#self.normScores + 1] = normScore
  end

  self:visualiseFilters()

  local avgV = self:validationStats()   -- I remembered there are some problems with this function invocation. But it is helpful.
  log.info('Average V: ' .. avgV)

  -- Save latest weights
  log.info('Saving weights')
  self:saveWeights('last')

  if valAvgScore > self.bestValScore then
    log.info('New best average score')
    self.bestValScore = valAvgScore
    self:saveWeights('best')
  end


  log.info('Saving weights on training step')
  local avs = string.format('%.5f', valAvgScore) or 'nvl'
  self:saveWeights(avs)

  if self.reportWeights then
    local reports = self:weightsReport()
    for r = 1, #reports do
      log.info(reports[r])
    end
  end
end

function ValidationAgent:saveWeights(name)
  log.info('Saving weights')
  torch.save(paths.concat('experiments', self._id, name..'.weights.t7'), self.theta)
end

-- Saves network convolutional filters as images
function ValidationAgent:visualiseFilters()
  local filters = self.model:getFilters()

  for i, v in ipairs(filters) do
    image.save(paths.concat('experiments', self._id, 'conv_layer_' .. i .. '.png'), v)
  end
end

local pprintArr = function(memo, v)
  return memo .. ', ' .. v
end

-- Reports absolute network weights and gradients
function ValidationAgent:weightsReport()
  -- Collect layer with weights
  local weightLayers = self.policyNet_:findModules('nn.SpatialConvolution')
  if #weightLayers == 0 then
    -- Assume cuDNN convolutions
    weightLayers = self.policyNet:findModules('cudnn.SpatialConvolution')
  end
  local fcLayers = self.policyNet_:findModules('nn.Linear')
  weightLayers = _.append(weightLayers, fcLayers)

  -- Array of norms and maxima
  local wNorms = {}
  local wMaxima = {}
  local wGradNorms = {}
  local wGradMaxima = {}

  -- Collect statistics
  for l = 1, #weightLayers do
    local w = weightLayers[l].weight:clone():abs() -- Weights (absolute)
    wNorms[#wNorms + 1] = torch.mean(w) -- Weight norms:
    wMaxima[#wMaxima + 1] = torch.max(w) -- Weight max
    w = weightLayers[l].gradWeight:clone():abs() -- Weight gradients (absolute)
    wGradNorms[#wGradNorms + 1] = torch.mean(w) -- Weight grad norms:
    wGradMaxima[#wGradMaxima + 1] = torch.max(w) -- Weight grad max
  end

  -- Create report string table
  local reports = {
    'Weight norms: ' .. _.reduce(wNorms, pprintArr),
    'Weight max: ' .. _.reduce(wMaxima, pprintArr),
    'Weight gradient norms: ' .. _.reduce(wGradNorms, pprintArr),
    'Weight gradient max: ' .. _.reduce(wGradMaxima, pprintArr)
  }

  return reports
end


function ValidationAgent:validationStats()
  local indices = torch.linspace(2, self.valSize+1, self.valSize):long() --local indices = self.valMemory:sample()  -- Should not sample here, especially for async recurrent models
  local states, actions, rewards, transitions, terminals = self.valMemory:retrieve(indices)

  local avgV
  if self.opt.recurrent then
    -- Async model with recurrent modules. In setup.lua, we forced histLen to be 1 when recurrent is true and async is utilized
    -- So, the returned transitions from valuMemory is a 5-dim tensor, whose size is like 50*1*1*24*24. 50 is size of transitions
    -- returned from experiment memory. 2nd dim 1 is histLen, 3rd dim 1 is number of channels of input data (1*24*24 is from Catch)
    transitions = transitions:squeeze(2)  -- squeeze the histLen dim (which has to be 1, according to restriction in setup.lua)
    if self.opt.actor_critic then
      local Vs = {}
      for i=1, transitions:size(1) do
        if terminals[i] > 0.5 then
          self.policyNet_:forget()
        else
          Vs[#Vs+1] = self.policyNet_:forward(transitions[i])[1]:squeeze()  -- actor-critic model has 2 outputs, 1st is a float number of V, 2nd is a 1-dim tensor of Q values
        end
      end
      avgV = torch.Tensor(Vs):mean()
    else
      local QPrimes = {}
      for i=1, transitions:size(1) do
        QPrimes[#QPrimes+1] = self.policyNet_:forward(transitions[i])
        if terminals[i] > 0.5 then
          self.policyNet_:forget()
        end
      end
      local _QPrimes = torch.Tensor(#QPrimes, table.unpack(QPrimes[1]:size():totable()))
      for i=1, #QPrimes do
        _QPrimes[i] = QPrimes[i]
      end
      QPrimes = _QPrimes
      local VPrime = torch.max(QPrimes, 3)  -- for QAgent, net:forward(states) has 2-dim output of size 1*10 (10 acts), right not QPrimes is 3-dim, with 1-dim be batch index. Oh oh, the 1 value might be # of head

      -- If it is CI data, pick up actions according to adpType
      if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
        VPrime = torch.min(QPrimes, 3)
        for ib=1, QPrimes:size(1) do  -- batch size
          if terminals[ib] < 0.5 then -- only need to calculate Q' for non-terminated next states
            local adpT = 0
            assert(transitions:dim() == 4, 'Dim of transitions should be 4, because we adopted squeeze()')
            if transitions[ib][1][1][-4] > 0.1 then adpT = 1 elseif transitions[ib][1][1][-3] > 0.1 then adpT = 2 elseif transitions[ib][1][1][-2] > 0.1 then adpT = 3 elseif transitions[ib][1][1][-1] > 0.1 then adpT = 4 end
            assert(adpT >=1 and adpT <= 4)
            for i=1, QPrimes:size(2) do    -- index of head in bootstraps in nn output
              for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
                if QPrimes[ib][i][j] >= VPrime[ib][i][1] then
                  VPrime[ib][i][1] = QPrimes[ib][i][j]
                end
              end
            end
          else
            for i=1, QPrimes:size(2) do
              VPrime[ib][i][1] = 0
            end
          end
        end
      end
      avgV = VPrime:mean()
    end
  else
    -- Async model without recurrent modules
    if self.opt.actor_critic then
      local Vs = self.policyNet_:forward(transitions)[1]  -- actor-critic model has 2 outputs, 1st is a float number of V, 2nd is a 1-dim tensor of Q values
      avgV = Vs:mean()
    else
      local QPrimes = self.policyNet_:forward(transitions) -- in real learning targetNet but doesnt matter for validation
      local VPrime = torch.max(QPrimes, 3)  -- for QAgent, net:forward(states) has 2-dim output of size 1*10 (10 acts), right not QPrimes is 3-dim, with 1-dim be batch index. Oh oh, the 1 value might be # of head

      -- If it is CI data, pick up actions according to adpType
      if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
        VPrime = torch.min(QPrimes, 3)
        for ib=1, QPrimes:size(1) do  -- batch size
          if terminals[ib] < 0.5 then -- only need to calculate Q' for non-terminated next states
            local adpT = 0
            if transitions[ib][-1][1][1][-4] > 0.1 then adpT = 1 elseif transitions[ib][-1][1][1][-3] > 0.1 then adpT = 2 elseif transitions[ib][-1][1][1][-2] > 0.1 then adpT = 3 elseif transitions[ib][-1][1][1][-1] > 0.1 then adpT = 4 end
            assert(adpT >=1 and adpT <= 4)
            for i=1, QPrimes:size(2) do    -- index of head in bootstraps in nn output
              for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
                if QPrimes[ib][i][j] >= VPrime[ib][i][1] then
                  VPrime[ib][i][1] = QPrimes[ib][i][j]
                end
              end
            end
          else
            for i=1, QPrimes:size(2) do
              VPrime[ib][i][1] = 0
            end
          end
        end
      end
      avgV = VPrime:mean()
    end
  end

  self.avgV[#self.avgV + 1] = avgV
  self:plotValidation()
  return avgV
end


function ValidationAgent:plotValidation()
  -- Plot and save losses
  if #self.losses > 0 then
    local losses = torch.Tensor(self.losses)
    gnuplot.pngfigure(paths.concat('experiments', self._id, 'losses.png'))
    gnuplot.plot('Loss', torch.linspace(math.floor(self.learnStart/self.progFreq), math.floor(self.globals.step/self.progFreq), #self.losses), losses, '-')
    gnuplot.xlabel('Step (x' .. self.progFreq .. ')')
    gnuplot.ylabel('Loss')
    gnuplot.plotflush()
    torch.save(paths.concat('experiments', self._id, 'losses.t7'), losses)
  end
  ---- Plot and save V
  --local epochIndices = torch.linspace(1, #self.avgV, #self.avgV)
  --local Vs = torch.Tensor(self.avgV)
  --gnuplot.pngfigure(paths.concat('experiments', self._id, 'Vs.png'))
  --gnuplot.plot('V', epochIndices, Vs, '-')
  --gnuplot.xlabel('Epoch')
  --gnuplot.ylabel('V')
  --gnuplot.movelegend('left', 'top')
  --gnuplot.plotflush()
  --torch.save(paths.concat('experiments', self._id, 'V.t7'), Vs)
  ---- Plot and save TD-error δ
  --if #self.avgTdErr>0 then
  --  local TDErrors = torch.Tensor(self.avgTdErr)
  --  gnuplot.pngfigure(paths.concat('experiments', self._id, 'TDErrors.png'))
  --  gnuplot.plot('TD-Error', epochIndices, TDErrors, '-')
  --  gnuplot.xlabel('Epoch')
  --  gnuplot.ylabel('TD-Error')
  --  gnuplot.plotflush()
  --  torch.save(paths.concat('experiments', self._id, 'TDErrors.t7'), TDErrors)
  --end
  -- Plot and save average score
  local scores = torch.Tensor(self.valScores)
  local epochIndices = torch.linspace(1, #self.valScores, #self.valScores)
  gnuplot.pngfigure(paths.concat('experiments', self._id, 'scores.png'))
  gnuplot.plot('Score', epochIndices, scores, '-')
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('Average Score')
  gnuplot.movelegend('left', 'top')
  gnuplot.plotflush()
  torch.save(paths.concat('experiments', self._id, 'scores.t7'), scores)
  --  -- Plot and save normalised score
  --if #self.normScores > 0 then
  --  local normScores = torch.Tensor(self.normScores)
  --  gnuplot.pngfigure(paths.concat('experiments', self._id, 'normScores.png'))
  --  gnuplot.plot('Score', epochIndices, normScores, '-')
  --  gnuplot.xlabel('Epoch')
  --  gnuplot.ylabel('Normalised Score')
  --  gnuplot.movelegend('left', 'top')
  --  gnuplot.plotflush()
  --  torch.save(paths.concat('experiments', self._id, 'normScores.t7'), normScores)
  --end
  gnuplot.close()
end


function ValidationAgent:evaluate(display)
  log.info('Evaluation mode')

  self.theta_:copy(self.theta)
  if self.lstm then self.lstm:forget() end

  self.stateBuffer:clear()
  self.env:evaluate()
  self.policyNet_:evaluate()

  local valEpisode = 1
  local valEpisodeScore = 0
  local valTotalScore = 0
  local valStep = 1

  local reward, terminal = 0, false
  local observation, adpType = self.env:start()   -- Todo: pwang8. This has been changed a little for compatibility with CI sim

  while valEpisode <= self.opt.evaTrajs do
    observation = self.model:preprocess(observation)
    if terminal then
      self.stateBuffer:clear()
    else
      self.stateBuffer:push(observation)
    end
    if not terminal then
      local state = self.stateBuffer:readAll()

      local action = self:selectAction(state)

      if self.opt.env == 'UserSimLearner/CIUserSimEnv' and self.opt.evalRand then -- right now, rand policy can only be evaluated in async mode
        local adpT = 0
        if state[-1][1][-4] > 0.1 then adpT = 1 elseif state[-1][1][-3] > 0.1 then adpT = 2 elseif state[-1][1][-2] > 0.1 then adpT = 3 elseif state[-1][1][-1] > 0.1 then adpT = 4 end
        assert(adpT >=1 and adpT <= 4)
        action = torch.random(self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2])
      end

      reward, observation, terminal, adpType = self.env:step(action - self.actionOffset)
      valEpisodeScore = valEpisodeScore + reward
    else
      if self.lstm then self.lstm:forget() end

      -- Print score every 10 episodes
      if valEpisode % 10 == 0 then
        local avgScore = valTotalScore/math.max(valEpisode - 1, 1)
        log.info('[VAL] Steps: ' .. valStep .. ' | Episode ' .. valEpisode
                .. ' | Score: ' .. valEpisodeScore .. ' | TotScore: ' .. valTotalScore .. ' | AvgScore: %.2f', avgScore)
      end

      -- Start a new episode
      valEpisode = valEpisode + 1
      reward, terminal = 0, false
      observation, adpType = self.env:start()    -- Todo: pwang8. This has been changed a little for compatibility with CI sim
      valTotalScore = valTotalScore + valEpisodeScore -- Only add to total score at end of episode
      valEpisodeScore = reward -- Reset episode score
    end
    valStep = valStep + 1
  end

  log.info('[VAL] Final evaluation avg score: ' .. valTotalScore/self.opt.evaTrajs)

  if display then
    display:createVideo()
  end
end


function ValidationAgent:ISevaluate(display)
  --- From Validation
  log.info('IS Evaluation mode')
  -- Set environment and agent to evaluation mode
  self.theta_:copy(self.theta)
  if self.lstm then self.lstm:forget() end

  self.stateBuffer:clear()
  self.env:evaluate()
  self.policyNet_:evaluate()

  local userSim = self.env.CIUSim

  local totalScoreIs = 0
  local sumOfProbWeights = 0
  local totalScoreIsDiscout = 0
  for uid, uRec in pairs(userSim.realUserRLTerms) do
    local rwd = userSim.realUserRLRewards[uid][1]
    local weight = 1.0
    local weightDiscount = 1.0
    for k, v in pairs(uRec) do
      local observation = self.model:preprocess(userSim.realUserRLStatePrepInd[uid][k])
      if v < 1 then -- not terminal
        self.stateBuffer:push(observation)
      else
        self.stateBuffer:clear()
      end
      if v < 1 then -- not terminal
        local state = self.stateBuffer:readAll()
        local _, actDist = self:selectAction(state)
        local randprob = 0.333333
        if userSim.realUserRLTypes[uid][k] == userSim.CIFr.ciAdp_BryceSymp or
                userSim.realUserRLTypes[uid][k] == userSim.CIFr.ciAdp_PresentQuiz then
          randprob = 0.5
        end
        weight = weight * (actDist[userSim.realUserRLActs[uid][k]] / randprob)
      else  -- terminal
        if self.lstm then self.lstm:forget() end
        sumOfProbWeights = sumOfProbWeights + weight  -- This is the sum of all trajectory appearance probabilities (no negative values)
        weight = weight * rwd -- rwd can be -1 or 1
        weightDiscount = weight * math.pow(self.opt.gamma, k-1)
      end
    end
    if self.opt.isevaprt then log.info('IS policy weigthed value,' .. weight .. ', discnt, ' .. weightDiscount) end
    totalScoreIs = totalScoreIs + weight
    totalScoreIsDiscout = totalScoreIsDiscout + weightDiscount
  end

  local trjCnt = TableSet.countsInSet(userSim.realUserRLTerms)
  log.info('Importance Sampling rewards on test set,' .. totalScoreIs/sumOfProbWeights .. ', total, ' .. totalScoreIs ..
      'Discount Importance Sampling rewards on test set, '.. totalScoreIsDiscout/sumOfProbWeights .. ', total, ' .. totalScoreIsDiscout)

end

return ValidationAgent
