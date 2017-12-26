local _ = require 'moses'
local class = require 'classic'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local Model = require 'Model'
local Experience = require 'Experience'
local CircularQueue = require 'structures/CircularQueue'
local Singleton = require 'structures/Singleton'
local AbstractAgent = require 'async/AbstractAgent'
require 'classic.torch' -- Enables serialisation
require 'modules/rmspropm' -- Add RMSProp with momentum

-- Detect QT for image display
local qt = pcall(require, 'qt')

local Agent = classic.class('Agent', AbstractAgent)
local TableSet = require 'MyMisc.TableSetMisc'

-- Creates a DQN agent
function Agent:_init(opt, envObj)
  -- Experiment ID
  self._id = opt._id
  self.experiments = opt.experiments
  -- Actions
  self.m = opt.actionSpec[3][2] - opt.actionSpec[3][1] + 1 -- Number of discrete actions
  self.actionOffset = 1 - opt.actionSpec[3][1] -- Calculate offset if first action is not indexed as 1

  -- Initialise model helper
  self.model = Model(opt)
  -- Create policy and target networks
  self.policyNet = self.model:create()
  self.targetNet = self.policyNet:clone() -- Create deep copy for target network
  self.targetNet:evaluate() -- Target network always in evaluation mode
  self.tau = opt.tau    -- Target network is updated every tau steps
  self.doubleQ = opt.doubleQ  -- A bool value, indicating whether to use Double-Q Learning
  -- Network parameters θ and gradients dθ
  self.theta, self.dTheta = self.policyNet:getParameters()

  -- Boostrapping
  self.bootstraps = opt.bootstraps  -- for the Catch demo, this bootstraps value is set to 0 in default
  self.head = 1 -- Identity of current episode bootstrap head
  self.heads = math.max(opt.bootstraps, 1) -- Number of heads

  -- Recurrency
  self.recurrent = opt.recurrent  -- bool. For the Catch demo, this recurrent default value is false
  self.histLen = opt.histLen  -- DQN standard is 4, DRQN is 10 (Comments from original authors)

  -- Reinforcement learning parameters
  self.gamma = opt.gamma  -- discount factor, 0.99 in default
  self.rewardClip = opt.rewardClip  -- 1 in default
  self.tdClip = opt.tdClip  -- 1 in default
  self.epsilonStart = opt.epsilonStart  -- initial epsilon value used in epdilon-greedy. Inital value is 1
  self.epsilonEnd = opt.epsilonEnd
  self.epsilonGrad = (opt.epsilonEnd - opt.epsilonStart)/opt.epsilonSteps -- Greediness ε decay factor
  self.PALpha = opt.PALpha  -- this value is a param used in Persistent Advantage Learning, the default value is 0.9

  -- State buffer
  self.stateBuffer = CircularQueue(opt.recurrent and 1 or opt.histLen, opt.Tensor, opt.stateSpec[2])  -- the buffer stores histLen long of observations
  -- Experience replay memory
  self.memory = Experience(opt.memSize, opt)
  self.memSampleFreq = opt.memSampleFreq  -- Freq to "learn"
  self.memNSamples = opt.memNSamples  -- Number of optimizations to take in each learning process
  self.memSize = opt.memSize
  self.memPriority = opt.memPriority  -- prioritized experience replay. Could be '', or 'rank' right now

  -- Training mode
  self.isTraining = false
  self.batchSize = opt.batchSize
  self.learnStart = opt.learnStart
  self.progFreq = opt.progFreq  -- progress reporting frequency
  self.gradClip = opt.gradClip
  -- Optimiser parameters
  self.optimiser = opt.optimiser
  self.optimParams = {
    learningRate = opt.eta,
    momentum = opt.momentum
  }

  -- Q-learning variables (per head)
  self.QPrimes = opt.Tensor(opt.batchSize, self.heads, self.m)
  self.tdErr = opt.Tensor(opt.batchSize, self.heads)
  self.VPrime = opt.Tensor(opt.batchSize, self.heads, 1)

  -- Validation variables
  self.valSize = opt.valSize
  self.valMemory = Experience(opt.valSize + 3, opt, true) -- Validation experience replay memory (with empty starting state...states...final transition...blank state)
  self.losses = {}
  self.avgV = {} -- Running average of V(s')
  self.avgTdErr = {} -- Running average of TD-error δ
  self.valScores = {} -- Validation scores (passed from main script)
  self.normScores = {} -- Normalised validation scores (passed from main script)

  -- Tensor creation
  self.Tensor = opt.Tensor

  -- Saliency display
  self:setSaliency(opt.saliency) -- Set saliency option on agent and model, in opt it can be <none>|normal|guided|deconvnet, then transferred to true/false
  if #opt.stateSpec[2] == 3 then -- Make salie0ncy map only for visual states
    self.saliencyMap = opt.Tensor(1, opt.stateSpec[2][2], opt.stateSpec[2][3]):zero()
    self.inputGrads = opt.Tensor(opt.histLen*opt.stateSpec[2][1], opt.stateSpec[2][2], opt.stateSpec[2][3]):zero() -- Gradients with respect to the input (for saliency maps) Todo:pwang8. Is this inputGrads of the correct size if recurrent is used?
  end

  -- Get singleton instance for step
  self.globals = Singleton.getInstance()  -- this global singleton stores one value, the steps
  self.opt = opt
  -- Sorry, adding ugly code here again, just for CI data compatability
  self.CIActAdpBound = {{1, 3}, {4, 5}, {6, 8}, {9, 10} }
  self.envRef = envObj -- This is a ref to the environment object. I used it to directly get access to raw human user data in RL agent training. Added on Mar 22, 2017

  -- The following variables are used in reconstructing rl states/acts/rewards/terminals in raw training set.
  -- The reason that these variables need to be reconstructed is that original variables in UserSimulator have
  -- non-numerical keys in tables, so not easy to iterate when sent to store in replay memory
  self.ruRLStates = {}
  self.ruRLActs = {}
  self.ruRLRewards = {}
  self.ruRLTerms = {}
  for kk, vv in pairs(self.envRef.CIUSim.realUserRLStatePrepInd) do
    for ik, iv in ipairs(vv) do
      self.ruRLStates[#self.ruRLStates+1] = iv:clone()
      if self.envRef.CIUSim.realUserRLTerms[kk][ik] > 0 then
        self.ruRLTerms[#self.ruRLStates] = true
      else
        self.ruRLTerms[#self.ruRLStates] = false
      end
      self.ruRLActs[#self.ruRLStates] = self.envRef.CIUSim.realUserRLActs[kk][ik]
      self.ruRLRewards[#self.ruRLStates] = self.envRef.CIUSim.realUserRLRewards[kk][ik]
      if not self.ruRLTerms[#self.ruRLTerms] then self.ruRLRewards[#self.ruRLRewards] = 0 end -- if not terminal, set reward to 0
    end
  end
  self.ruRLItemCnt = #self.ruRLStates

end

-- Sets training mode
function Agent:training()
  self.isTraining = true
  self.policyNet:training()
  -- Clear state buffer
  self.stateBuffer:clear()
  -- Reset bootstrap head
  if self.bootstraps > 0 then
    self.head = torch.random(self.bootstraps) -- here this self.bootstraps is a integer
  end
  -- Forget last sequence
  if self.recurrent then
    self.policyNet:forget()
    self.targetNet:forget()
  end
end

-- Sets evaluation mode
function Agent:evaluate()
  self.isTraining = false
  self.policyNet:evaluate()
  -- Clear state buffer
  self.stateBuffer:clear()
  -- Set previously stored state as invalid (as no transition stored)
  self.memory:setInvalid()
--  self.valMemory:setInvalid() -- This should not be necessary if validation starts later than the fullfilling of valmemory
  -- Reset bootstrap head
  if self.bootstraps > 0 then
    self.head = torch.random(self.bootstraps)
  end
  -- Forget last sequence
  if self.recurrent then
    self.policyNet:forget()
  end
end
  
-- Observes the results of the previous transition and chooses the next action to perform
function Agent:observe(reward, rawObservation, terminal)
  -- Clip reward for stability
  if self.rewardClip > 0 then
    reward = math.max(reward, -self.rewardClip)
    reward = math.min(reward, self.rewardClip)
  end

  -- Process observation of current state
  local observation = self.model:preprocess(rawObservation) -- Must avoid side-effects on observation from env

  -- Store in buffer depending on terminal status
  if terminal then
    self.stateBuffer:pushReset(observation) -- Will clear buffer on next push
  else
    self.stateBuffer:push(observation)  -- the size/capacity of the CircularQueue equals histLen
  end
  -- Retrieve current and historical states from state buffer
  local state = self.stateBuffer:readAll()  -- the returned value is one tensor containing all histLen frames -- state dim is 4*1*25 for CI data. 4 is histLen

  -- Set ε based on training vs. evaluation mode
  local epsilon = 0.001 -- Taken from tuned DDQN evaluation
  if self.isTraining then
    if self.globals.step < self.learnStart then
      -- Keep ε constant before learning starts
      epsilon = self.epsilonStart
    else
      -- Use annealing ε
      epsilon = math.max(self.epsilonStart + (self.globals.step - self.learnStart - 1)*self.epsilonGrad, self.epsilonEnd)
    end
  end

  local actDist = {}  -- Todo: pwang8. Check correctness. This is act selection probability dist (could be used in Importance sampling). Only used in evaluation
  local aIndex = 1 -- In a terminal state, choose no-op/first action by default
  if not terminal then
    if not self.isTraining and self.bootstraps > 0 then
      -- Retrieve estimates from all heads
      local QHeads = self.policyNet:forward(state)  -- QHeads (2-dim) has size (heads_count * action_num)

      -- Calculate the sum of each action's Q values over all heads
      local actQValSumHeads = QHeads:sum(1):squeeze()

      -- Use ensemble policy with bootstrap heads (in evaluation mode)
      local QHeadsMax, QHeadsMaxInds
      -- If it is CI data, pick up actions according to adpType
      if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
        QHeadsMax, QHeadsMaxInds = QHeads:min(2)
        local adpT = 0
        if state[-1][1][-4] > 0.1 then adpT = 1 elseif state[-1][1][-3] > 0.1 then adpT = 2 elseif state[-1][1][-2] > 0.1 then adpT = 3 elseif state[-1][1][-1] > 0.1 then adpT = 4 end
        assert(adpT >=1 and adpT <= 4)
        for i=1, QHeads:size(1) do
          for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
            if QHeads[i][j] >= QHeadsMax[i][1] then
              QHeadsMax[i][1] = QHeads[i][j]
              QHeadsMaxInds[i][1] = j
            end
          end
        end

        -- Calculate the action selection distribution
        local temQsum = 0
        for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
          temQsum = temQsum + math.exp(self.opt.actDistT * actQValSumHeads[j])
        end
        for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
          actDist[j] = math.exp(self.opt.actDistT * actQValSumHeads[j]) / temQsum
        end
      else
        QHeadsMax, QHeadsMaxInds = QHeads:max(2) -- Find max action per head -- torch.mode() is a function returns most frequently appeared element (maths)
      end

      aIndex = torch.mode(QHeadsMaxInds:float(), 1)[1][1] -- TODO: Torch.CudaTensor:mode is missing

      -- Plot uncertainty in ensemble policy
      if qt then
        gnuplot.hist(QHeadsMaxInds, self.m, 0.5, self.m + 0.5)
      end

      -- Compute saliency map
      if self.saliency then
        self:computeSaliency(state, aIndex, true)
      end

    elseif torch.uniform() < epsilon then 
      -- Choose action by ε-greedy exploration (even with bootstraps)
      aIndex = torch.random(1, self.m)

      -- Retrieve estimates from all heads
      local QHeads = self.policyNet:forward(state)  -- QHeads (2-dim) has size (heads_count * action_num). We call the forward() bcz we calculate actDist. Also it is necessary to call forward when recurrent is utilized
      -- Calculate the sum of each action's Q values over all heads
      local actQValSumHeads = QHeads:sum(1):squeeze()

      -- If it is CI data, pick up actions according to adpType
      if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
        local adpT = 0
        if state[-1][1][-4] > 0.1 then adpT = 1 elseif state[-1][1][-3] > 0.1 then adpT = 2 elseif state[-1][1][-2] > 0.1 then adpT = 3 elseif state[-1][1][-1] > 0.1 then adpT = 4 end
        assert(adpT >=1 and adpT <= 4)
        aIndex = torch.random(self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2])

        -- Calculate the action selection distribution
        local temQsum = 0
        for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
          temQsum = temQsum + math.exp(self.opt.actDistT * actQValSumHeads[j])
        end
        for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
          actDist[j] = math.exp(self.opt.actDistT * actQValSumHeads[j]) / temQsum
        end
      end

      -- Reset saliency if action not chosen by network
      if self.saliency then
        self.saliencyMap:zero()
      end

    else
      -- Retrieve estimates from all heads
      local QHeads = self.policyNet:forward(state)

      -- Sample from current episode head (indexes on first dimension with no batch)
      local Qs = QHeads:select(1, self.head)  -- This self.head value is randomly set when Agent:training() is called
      local maxQ = Qs[1]                      --
      local bestAs = {1}

      -- If it is CI data, pick up actions according to adpType
      if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
        local adpT = 0
        if state[-1][1][-4] > 0.1 then adpT = 1 elseif state[-1][1][-3] > 0.1 then adpT = 2 elseif state[-1][1][-2] > 0.1 then adpT = 3 elseif state[-1][1][-1] > 0.1 then adpT = 4 end
        assert(adpT >=1 and adpT <= 4)
        maxQ = Qs[self.CIActAdpBound[adpT][1]]
        bestAs = {self.CIActAdpBound[adpT][1]}
        for a = self.CIActAdpBound[adpT][1]+1, self.CIActAdpBound[adpT][2] do
          if Qs[a] > maxQ then
            maxQ = Qs[a]
            bestAs = {a}
          elseif Qs[a] == maxQ then -- Ties can occur even with floats
            bestAs[#bestAs + 1] = a
          end
        end

        -- Calculate the action selection distribution
        local temQsum = 0
        for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
          temQsum = temQsum + math.exp(self.opt.actDistT * Qs[j])
        end
        for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
          actDist[j] = math.exp(self.opt.actDistT * Qs[j]) / temQsum
        end

      else
        -- Find best actions
        for a = 2, self.m do
          if Qs[a] > maxQ then
            maxQ = Qs[a]
            bestAs = {a}
          elseif Qs[a] == maxQ then -- Ties can occur even with floats
            bestAs[#bestAs + 1] = a
          end
        end
      end

      -- Perform random tie-breaking (if more than one argmax action)
      aIndex = bestAs[torch.random(1, #bestAs)]

      -- Compute saliency
      if self.saliency then
        self:computeSaliency(state, aIndex, false)
      end
    end
  end

  local orig_terminal = terminal  -- This is only used for setting forget() for recurrent net at the bottom of this block
  local orig_act = aIndex

  -- If training
  if self.isTraining then
    -- if train with raw data from log files
    if self.opt.trainWithRawData then   -- The change here is that, feed raw player data into experience replay memory
      local iter_raw = self.globals.step % self.ruRLItemCnt + 1
      reward = self.ruRLRewards[iter_raw]
      observation = self.ruRLStates[iter_raw]   -- The preprocessing is not necessary for CI, since its input is not image
      terminal = self.ruRLTerms[iter_raw]
      aIndex = self.ruRLActs[iter_raw]
    end

    -- Store experience tuple parts (including pre-emptive action)
    -- Attention: state is the input into NN, which contains histLen observation steps. When storing observation into
    -- experience memory, observation is directly used. This can aviod duplicated observation stored in memory. By pwang8.
    self.memory:store(reward, observation, terminal, aIndex) -- TODO: Sample independent Bernoulli(p) bootstrap masks for all heads; p = 1 means no masks needed

    --- Todo: pwang8. test
    if self.opt.env == 'UserSimLearner/CIUserSimEnv' and observation[1][1][-4] < 1 and observation[1][1][-3] < 1 and
            observation[1][1][-2] < 1 and observation[1][1][-1] < 1 and not terminal then
      print('Error ===========', observation, 'act:', aIndex, 'ter:', terminal)
      os.exit()
    end

    -- Collect validation transitions at the start
    if self.globals.step <= self.valSize + 1 then
      self.valMemory:store(reward, observation, terminal, aIndex)
    end

    -- Sample uniformly or with prioritised sampling
    if self.globals.step % self.memSampleFreq == 0 and self.globals.step >= self.learnStart then
      for n = 1, self.memNSamples do
        -- Optimise (learn) from experience tuples
        self:optimise(self.memory:sample())
      end
    end

    -- Update target network every τ steps
    if self.globals.step % self.tau == 0 and self.globals.step >= self.learnStart then
      self.targetNet = self.policyNet:clone()
      self.targetNet:evaluate()
    end

    -- Rebalance priority queue for prioritised experience replay
    if self.globals.step % self.memSize == 0 and self.memPriority then  -- self.memSize is experience replay memory size, memPriority is type of PER
      self.memory:rebalance()
    end
  end

  if orig_terminal then -- terminal then, orig_terminal is used because terminal is changed if raw data is used in training
    if self.bootstraps > 0 then
      -- Change bootstrap head for next episode
      self.head = torch.random(self.bootstraps)
    end
    if self.recurrent then
      -- Forget last sequence
      self.policyNet:forget()
    end
  end

  -- Return action index with offset applied
  return orig_act - self.actionOffset, actDist
end

-- Learns from experience
function Agent:learn(x, indices, ISWeights, isValidation)
  -- Copy x to parameters θ if necessary
  if x ~= self.theta then
    self.theta:copy(x)
  end
  -- Reset gradients dθ
  self.dTheta:zero()

  -- Retrieve experience tuples
  local memory = isValidation and self.valMemory or self.memory
  local states, actions, rewards, transitions, terminals = memory:retrieve(indices) -- Terminal status is for transition (can't act in terminal state)
  local N = actions:size(1) -- # of s-a-s transitions (batchSize)
  -- size of states/transitions is 32*10*1*24*24 for Catch demo, when recurrent is true and histLen is 10. batchSize is 32.
  if self.recurrent then
    -- Forget last sequence
    self.policyNet:forget()
    self.targetNet:forget()
  end
  -- Dim of transitions is (32*4*1*1*25) for CI sim. 32 is batch size, 4 is histLen.

  -- Perform argmax action selection
  local APrimeMax, APrimeMaxInds
  if self.doubleQ then
    -- Calculate Q-values from transition using policy network
    self.QPrimes = self.policyNet:forward(transitions) -- Find argmax actions using policy network
    -- Dim of self.QPrimes is 32*5*10. 32 is batchSize, 5 is boostraps #, 10 is output (action) dim. If bootstraps is set to 0, it becomes 32*1*10
    -- Perform argmax action selection on transition using policy network: argmax_a[Q(s', a; θpolicy)]
    APrimeMax, APrimeMaxInds = torch.max(self.QPrimes, 3)

    -- If it is CI data, pick up actions according to adpType
    if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
      APrimeMax, APrimeMaxInds = torch.min(self.QPrimes, 3)
      for ib=1, N do  -- batch size
        if terminals[ib] < 1 then -- only need to calculate Q' for non-terminated next states
          local adpT = 0
          if transitions[ib][-1][1][1][-4] > 0.1 then adpT = 1 elseif transitions[ib][-1][1][1][-3] > 0.1 then adpT = 2 elseif transitions[ib][-1][1][1][-2] > 0.1 then adpT = 3 elseif transitions[ib][-1][1][1][-1] > 0.1 then adpT = 4 end
          assert(adpT >=1 and adpT <= 4)
          for i=1, self.QPrimes:size(2) do    -- index of head in bootstraps in nn output
            for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
              if self.QPrimes[ib][i][j] >= APrimeMax[ib][i][1] then
                APrimeMax[ib][i][1] = self.QPrimes[ib][i][j]
                APrimeMaxInds[ib][i][1] = j
              end
            end
          end
        end
      end
    end

    -- Calculate Q-values from transition using target network
    self.QPrimes = self.targetNet:forward(transitions) -- Evaluate Q-values of argmax actions using target network
  else
    -- Calculate Q-values from transition using target network
    self.QPrimes = self.targetNet:forward(transitions) -- Find and evaluate Q-values of argmax actions using target network
    -- Perform argmax action selection on transition using target network: argmax_a[Q(s', a; θtarget)]
    APrimeMax, APrimeMaxInds = torch.max(self.QPrimes, 3)

    -- If it is CI data, pick up actions according to adpType
    if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
      APrimeMax, APrimeMaxInds = torch.min(self.QPrimes, 3)
      for ib=1, N do  -- batch size
        if terminals[ib] < 1 then -- only need to calculate Q' for non-terminated next states
          local adpT = 0
          if transitions[ib][-1][1][1][-4] > 0.1 then adpT = 1 elseif transitions[ib][-1][1][1][-3] > 0.1 then adpT = 2 elseif transitions[ib][-1][1][1][-2] > 0.1 then adpT = 3 elseif transitions[ib][-1][1][1][-1] > 0.1 then adpT = 4 end
          assert(adpT >=1 and adpT <= 4)
          for i=1, self.QPrimes:size(2) do    -- index of head in bootstraps in nn output
            for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
              if self.QPrimes[ib][i][j] >= APrimeMax[ib][i][1] then
                APrimeMax[ib][i][1] = self.QPrimes[ib][i][j]
                APrimeMaxInds[ib][i][1] = j
              end
            end
          end
        end
      end
    end

  end
  -- self.QPrimes is a 3-dim tensor. 1st dim is batch index, 2nd dim is head index in bootstarps, 3rd dim is action index
  -- Initially set target Y = Q(s', argmax_a[Q(s', a; θ)]; θtarget), where initial θ is either θtarget (DQN) or θpolicy (DDQN)
  local Y = self.Tensor(N, self.heads)
  for n = 1, N do
    self.QPrimes[n]:mul(1 - terminals[n]) -- Zero Q(s' a) when s' is terminal
    Y[n] = self.QPrimes[n]:gather(2, APrimeMaxInds[n])
  end
  -- Calculate target Y := r + γ.Q(s', argmax_a[Q(s', a; θ)]; θtarget)  -- rewards has dim 32 (1-dim), which is batch size, terminal has same dim
  Y:mul(self.gamma):add(rewards:repeatTensor(1, self.heads))

  -- Get all predicted Q-values from the current state
  if self.recurrent and self.doubleQ then -- call forget here since if doubleQ is used, policyNet has been utilized above in QPrime calc
    self.policyNet:forget()
  end
  local QCurr = self.policyNet:forward(states) -- Correct internal state of policy network before backprop.
  local QTaken = self.Tensor(N, self.heads)    -- QCurr of dim 32*7*10 in CI sim, with 32 batchSize, 7 heads(bootstraps), 10 actions
  -- Get prediction of current Q-values with given actions
  for n = 1, N do
    QTaken[n] = QCurr[n][{{}, {actions[n]}}]  -- in QCurr[n], data are of 2-dim. 1st is head index, 2nd is action index.
  end

  -- Calculate TD-errors δ := ∆Q(s, a) = Y − Q(s, a)
  self.tdErr = Y - QTaken -- self.tdErr should be a 3-dim tensor. 1st dim is batch index, 2nd dim is head index, 3rd dim has only one value.

  -- Calculate Persistant Advantage Learning update(s)
  if self.PALpha > 0 then
    -- Calculate Q(s, a) and V(s) using target network
    if self.recurrent then
      self.targetNet:forget()
    end
    local Qs = self.targetNet:forward(states) -- For CI sim, Qs of dim 32*7*10 in CI sim, with 32 batchSize, 7 heads(bootstraps), 10 actions
    local Q = self.Tensor(N, self.heads)
    for n = 1, N do
      Q[n] = Qs[n][{{}, {actions[n]}}]
    end
    local V = torch.max(Qs, 3) -- Current states cannot be terminal. 3-dim includes batchIndex, head index, action index

    -- If it is CI data, pick up actions according to adpType
    if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
      V = torch.min(Qs, 3)
      for ib=1, N do  -- batch size
          local adpT = 0
          if states[ib][-1][1][1][-4] > 0.1 then adpT = 1 elseif states[ib][-1][1][1][-3] > 0.1 then adpT = 2 elseif states[ib][-1][1][1][-2] > 0.1 then adpT = 3 elseif states[ib][-1][1][1][-1] > 0.1 then adpT = 4 end
          assert(adpT >=1 and adpT <= 4)
          for i=1, Q:size(2) do    -- index of head in bootstraps in nn output
            for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
              if Qs[ib][i][j] >= V[ib][i][1] then
                V[ib][i][1] = Qs[ib][i][j]
              end
            end
          end
      end
    end

    -- Calculate Advantage Learning update ∆ALQ(s, a) := δ − αPAL(V(s) − Q(s, a))
    local tdErrAL = self.tdErr - V:csub(Q):mul(self.PALpha)

    -- Calculate Q(s', a) and V(s') using target network
    local QPrime = self.Tensor(N, self.heads)
    for n = 1, N do
      QPrime[n] = self.QPrimes[n][{{}, {actions[n]}}]
    end
    -- QPrime has dim of 32*7, with batchSize 32, and bootstraps 7
    self.VPrime = torch.max(self.QPrimes, 3)

    -- Attention: in CI sim environment, since actions are restricted by adpType, so intuitively PAL updates can not be applied
    if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
      for ib=1, N do  -- batch size
        if terminals[ib] < 1 then -- only need to calculate Q' for non-terminated next states
          local adpT = 0
          if transitions[ib][-1][1][1][-4] > 0.1 then adpT = 1 elseif transitions[ib][-1][1][1][-3] > 0.1 then adpT = 2 elseif transitions[ib][-1][1][1][-2] > 0.1 then adpT = 3 elseif transitions[ib][-1][1][1][-1] > 0.1 then adpT = 4 end
          assert(adpT >=1 and adpT <= 4)
          for i=1, self.QPrimes:size(2) do    -- index of head in bootstraps in nn output
            for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
              if self.QPrimes[ib][i][j] >= self.VPrime[ib][i][1] then
                self.VPrime[ib][i][1] = self.QPrimes[ib][i][j]
              end
            end
          end
        end
      end
      QPrime = self.VPrime:clone()
      -- Since in CI environment, actions are restricted with its belonged adpType,
      -- I just get rid of this αPAL(V(s') − Q(s', a)) calculation, whose original purpose is to encourage repeating recent actions.
    end

    -- Calculate Persistent Advantage Learning update ∆PALQ(s, a) := max[∆ALQ(s, a), δ − αPAL(V(s') − Q(s', a))]
    self.tdErr = torch.max(torch.cat(tdErrAL, self.tdErr:csub((self.VPrime:csub(QPrime):mul(self.PALpha))), 3), 3):view(N, self.heads, 1)
  else  -- when self.PALpha <= 0
    -- Todo: pwang8. This is only needed for CI sim. We are going to update self.VPrime for purpose of validation
    -- QPrime has dim of 32*7, with batchSize 32, and bootstraps 7
    self.VPrime = torch.max(self.QPrimes, 3)

    -- Attention: in CI sim environment, since actions are restricted by adpType, so intuitively PAL updates can not be applied
    if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
      for ib=1, N do  -- batch size
        if terminals[ib] < 1 then -- only need to calculate Q' for non-terminated next states
          local adpT = 0
          if transitions[ib][-1][1][1][-4] > 0.1 then adpT = 1 elseif transitions[ib][-1][1][1][-3] > 0.1 then adpT = 2 elseif transitions[ib][-1][1][1][-2] > 0.1 then adpT = 3 elseif transitions[ib][-1][1][1][-1] > 0.1 then adpT = 4 end
          assert(adpT >=1 and adpT <= 4)
          for i=1, self.QPrimes:size(2) do    -- index of head in bootstraps in nn output
            for j=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
              if self.QPrimes[ib][i][j] >= self.VPrime[ib][i][1] then
                self.VPrime[ib][i][1] = self.QPrimes[ib][i][j]
              end
            end
          end
        end
      end
    end

  end

  -- Calculate loss
  local loss
  if self.tdClip > 0 then
    -- Squared loss is used within clipping range, absolute loss is used outside (approximates Huber loss)
    local sqLoss = torch.cmin(torch.abs(self.tdErr), self.tdClip)
    local absLoss = torch.abs(self.tdErr) - sqLoss
    loss = torch.mean(sqLoss:pow(2):mul(0.5):add(absLoss:mul(self.tdClip))) -- Average over heads

    -- Clip TD-errors δ
    self.tdErr:clamp(-self.tdClip, self.tdClip)
  else
    -- Squared loss
    loss = torch.mean(self.tdErr:clone():pow(2):mul(0.5)) -- Average over heads
  end

  -- Exit if being used for validation metrics
  if isValidation then
    return
  end

  -- Send TD-errors δ to be used as priorities
  self.memory:updatePriorities(indices, torch.mean(self.tdErr, 2)) -- Use average error over heads
  -- Zero QCurr outputs (no error)
  QCurr:zero()
  -- Set TD-errors δ with given actions
  for n = 1, N do
    -- Correct prioritisation bias with importance-sampling weights
    QCurr[n][{{}, {actions[n]}}] = torch.mul(-self.tdErr[n], ISWeights[n]) -- Negate target to use gradient descent (not ascent) optimisers
  end

  -- Backpropagate (network accumulates gradients internally)
  self.policyNet:backward(states, QCurr) -- TODO: Work out why DRQN crashes on different batch sizes
  -- Clip the L2 norm of the gradients
  if self.gradClip > 0 then
    self.policyNet:gradParamClip(self.gradClip)
  end

  if self.recurrent then
    -- Forget last sequence
    self.policyNet:forget()
    self.targetNet:forget()
    -- Previous hidden state of policy net not restored as model parameters changed
  end

  return loss, self.dTheta
end

-- Optimises the network parameters θ
function Agent:optimise(indices, ISWeights)
  -- Create function to evaluate given parameters x
  local feval = function(x)
    return self:learn(x, indices, ISWeights)
  end
  
  -- Optimise
  local __, loss = optim[self.optimiser](feval, self.theta, self.optimParams)
  -- Store loss
  if self.globals.step % self.progFreq == 0 then
    self.losses[#self.losses + 1] = loss[1]
  end

  return loss[1]
end

-- Pretty prints array
local pprintArr = function(memo, v)
  return memo .. ', ' .. v
end

-- Reports absolute network weights and gradients
function Agent:report()
  -- Collect layer with weights
  local weightLayers = self.policyNet:findModules('nn.SpatialConvolution')
  if #weightLayers == 0 then
    -- Assume cuDNN convolutions
    weightLayers = self.policyNet:findModules('cudnn.SpatialConvolution')
  end
  local fcLayers = self.policyNet:findModules('nn.Linear')
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

-- Reports stats for validation
function Agent:validate()
  -- Validation variables
  local totalV, totalTdErr = 0, 0

  -- Loop over validation transitions
  local nBatches = math.ceil(self.valSize / self.batchSize)
  local ISWeights = self.Tensor(self.batchSize):fill(1)
  local startIndex, endIndex, batchSize, indices
  for n = 1, nBatches do
    startIndex = (n - 1)*self.batchSize + 2
    endIndex = math.min(n*self.batchSize + 1, self.valSize + 1)
    batchSize = endIndex - startIndex + 1
    indices = self.valMemory:sample()  --torch.linspace(startIndex, endIndex, batchSize):long()  -- This is a way to generate a tensor of # numbers ranging from startIn to endIn

    -- Perform "learning" (without optimisation)
    self:learn(self.theta, indices, ISWeights:narrow(1, 1, batchSize), true)
    -- tensor:narrow() returns the ref to the original tensor along dim 1, with indices ranging from 1 to batchSize. A little bit like select()

--    --- Calculate V(s') and TD-error δ -- For CI sim, this has been moved into learn()
--    if self.PALpha == 0 then
--      self.VPrime = torch.max(self.QPrimes, 3)
--    end
    -- Average over heads
    totalV = totalV + torch.mean(self.VPrime, 2):sum()
    totalTdErr = totalTdErr + torch.mean(self.tdErr, 2):abs():sum()
  end

  -- Average and insert values
  self.avgV[#self.avgV + 1] = totalV / self.valSize
  self.avgTdErr[#self.avgTdErr + 1] = totalTdErr / self.valSize

  -- Plot and save losses
  if #self.losses > 0 then
    local losses = torch.Tensor(self.losses)
    gnuplot.pngfigure(paths.concat(self.experiments, self._id, 'losses.png'))
    gnuplot.plot('Loss', torch.linspace(math.floor(self.learnStart/self.progFreq), math.floor(self.globals.step/self.progFreq), #self.losses), losses, '-')
    gnuplot.xlabel('Step (x' .. self.progFreq .. ')')
    gnuplot.ylabel('Loss')
    gnuplot.plotflush()
    torch.save(paths.concat(self.experiments, self._id, 'losses.t7'), losses)
  end
  -- Plot and save V
  local epochIndices = torch.linspace(1, #self.avgV, #self.avgV)
  local Vs = torch.Tensor(self.avgV)
  gnuplot.pngfigure(paths.concat(self.experiments, self._id, 'Vs.png'))
  gnuplot.plot('V', epochIndices, Vs, '-')
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('V')
  gnuplot.movelegend('left', 'top')
  gnuplot.plotflush()
  torch.save(paths.concat(self.experiments, self._id, 'V.t7'), Vs)
  -- Plot and save TD-error δ
  local TDErrors = torch.Tensor(self.avgTdErr)
  gnuplot.pngfigure(paths.concat(self.experiments, self._id, 'TDErrors.png'))
  gnuplot.plot('TD-Error', epochIndices, TDErrors, '-')
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('TD-Error')
  gnuplot.plotflush()
  torch.save(paths.concat(self.experiments, self._id, 'TDErrors.t7'), TDErrors)
  -- Plot and save average score
  local scores = torch.Tensor(self.valScores)
  gnuplot.pngfigure(paths.concat(self.experiments, self._id, 'scores.png'))
  gnuplot.plot('Score', epochIndices, scores, '-')
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('Average Score')
  gnuplot.movelegend('left', 'top')
  gnuplot.plotflush()
  torch.save(paths.concat(self.experiments, self._id, 'scores.t7'), scores)
    -- Plot and save normalised score
  if #self.normScores > 0 then
    local normScores = torch.Tensor(self.normScores)
    gnuplot.pngfigure(paths.concat(self.experiments, self._id, 'normScores.png'))
    gnuplot.plot('Score', epochIndices, normScores, '-')
    gnuplot.xlabel('Epoch')
    gnuplot.ylabel('Normalised Score')
    gnuplot.movelegend('left', 'top')
    gnuplot.plotflush()
    torch.save(paths.concat(self.experiments, self._id, 'normScores.t7'), normScores)
  end
  gnuplot.close()

  return self.avgV[#self.avgV], self.avgTdErr[#self.avgTdErr]
end

-- Saves network convolutional filters as images
function Agent:visualiseFilters()
  local filters = self.model:getFilters()

  for i, v in ipairs(filters) do
    image.save(paths.concat(self.experiments, self._id, 'conv_layer_' .. i .. '.png'), v)
  end
end

-- Sets saliency style
function Agent:setSaliency(saliency)
  self.saliency = saliency
  self.model:setSaliency(saliency)
end

-- Computes a saliency map (assuming a forward pass of a single state)
function Agent:computeSaliency(state, index, ensemble)
  -- Switch to possibly special backpropagation
  self.model:salientBackprop()

  -- Create artificial high target
  local maxTarget = self.Tensor(self.heads, self.m):zero()
  if ensemble then
    -- Set target on all heads (when using ensemble policy)
    maxTarget[{{}, {index}}] = 1
  else
    -- Set target on current head
    maxTarget[self.head][index] = 1
  end

  -- Backpropagate to inputs
  self.inputGrads = self.policyNet:backward(state, maxTarget)
  -- Saliency map ref used by Display
  self.saliencyMap = torch.abs(self.inputGrads:select(1, self.recurrent and 1 or self.histLen):float())

  -- Switch back to normal backpropagation
  self.model:normalBackprop()
end

-- Saves the network parameters θ
function Agent:saveWeights(path)
  torch.save(path, self.theta:float()) -- Do not save as CudaTensor to increase compatibility
end

-- Loads network parameters θ
function Agent:loadWeights(path)
  local weights = torch.load(path)
  self.theta:copy(weights)
  self.targetNet = self.policyNet:clone()
  self.targetNet:evaluate()
end

return Agent
