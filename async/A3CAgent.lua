local classic = require 'classic'
local optim = require 'optim'
local AsyncAgent = require 'async/AsyncAgent'
require 'modules/sharedRmsProp'
local OptimMisc = require 'MyMisc.OptimMisc'

local A3CAgent,super = classic.class('A3CAgent', 'AsyncAgent')

local TINY_EPSILON = 1e-6

function A3CAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)

  log.info('creating A3CAgent')

  self.policyNet_ = policyNet:clone()

  self.theta_, self.dTheta_ = self.policyNet_:getParameters()
  self.dTheta_:zero()

  self.policyTarget = self.Tensor(self.m)
  self.vTarget = self.Tensor(1)
  self.targets = { self.vTarget, self.policyTarget }

  self.rewards = torch.Tensor(self.batchSize)
  self.actions = torch.ByteTensor(self.batchSize)
  self.states = torch.Tensor(0)
  self.terminal_masks = torch.Tensor(self.batchSize+1):fill(1)  -- this value is 0 if terminal, 1 if not. In s1-a-s2, ter_m[t1] represents whether s1 is terminal
  self.tdReturns = torch.Tensor(self.batchSize+1):zero()   -- stores td returns used in accumulateGradients()
  self.beta = opt.entropyBeta

  self.env:training()

  self.opt = opt
  -- Sorry, adding ugly code here again, just for CI data compatability
  self.CIActAdpBound = {{1, 3}, {4, 5}, {6, 8}, {9, 10}}

  classic.strict(self)
end


function A3CAgent:learn(steps, from)
  self.step = from or 0
  self.stateBuffer:clear()
  local _trainEpisode = 0

  log.info('A3CAgent starting | steps=%d', steps)
  local reward, terminal, state = self:start()
  assert(not terminal, 'Starting state should not be terminal state')
  self.policyNet_:training()
  if self.opt.recurrent then self.policyNet_:forget() end

  self.states:resize(self.batchSize, table.unpack(state:size():totable()))

  self.tic = torch.tic()
  repeat
    self.theta_:copy(self.theta)
    self.terminal_masks[1] = terminal and 0 or 1    -- this variable indicates whether the state at batchIdx is terminal
    self.batchIdx = 0

    repeat
      self.batchIdx = self.batchIdx + 1
      self.states[self.batchIdx]:copy(state)

      local action = 1
      reward = 0

      assert(not terminal, 'Terminal state should not be observed here')
      action = self:probabilisticAction(state)
      reward, terminal, state = self:takeAction(action)
      self.actions[self.batchIdx] = action
      self.rewards[self.batchIdx] = reward

      if terminal then
        self.terminal_masks[self.batchIdx+1] = 0
        if self.batchIdx < self.batchSize then
          self.states[self.batchIdx+1]:copy(state)  -- dumb place holder for ending/terminal state
          self.actions[self.batchIdx+1] = 1         -- dumb place holder for ending/terminal state
          self.rewards[self.batchIdx+1] = 0         -- dumb place holder for ending/terminal state
          self.batchIdx = self.batchIdx+1
          reward, terminal, state = self:start()
          assert(not terminal, 'Starting state should not be terminal state')
          if self.opt.recurrent then self.policyNet_:forget() end
        end
      else
        self.terminal_masks[self.batchIdx+1] = 1    -- this variable indicates whether the state at batchIdx is terminal
        -- the indices for rewards and terminal_masks are different. For an s1-a-s2 transition, rewards[t1] represents
        -- the reward got in this transition. But terminal_masks[t1] means whether s1 is a terminal state (0 if it is terminal).
      end

      self:progress(steps)
    until self.batchIdx == self.batchSize

    _trainEpisode = _trainEpisode + 1   -- counter of training episodes
    self:accumulateGradients(terminal, state)

    -- Control param updating frequency. It performs as controlling batch size in training
    if _trainEpisode % self.opt.asyncOptimFreq == 0 then
      self:applyGradients(self.policyNet_, self.dTheta_, self.theta)
    end

    if terminal then
      reward, terminal, state = self:start()
      assert(not terminal, 'Starting state should not be terminal state')
      if self.opt.recurrent then self.policyNet_:forget() end
    end
  until self.step >= steps

  log.info('A3CAgent ended learning steps=%d', steps)
end


function A3CAgent:accumulateGradients(terminal, state)
  -- Now it is implemented in the forward updating way. This helps the training of FastLSTM module if the
  -- actor-critic model includes one. Jan 1, 2018

  -- calculate TD-return
  self.tdReturns:zero()
  if not terminal then
    self.tdReturns[self.batchIdx+1] = self.policyNet_:forward(state)[1]
  end
  for i=self.batchIdx,1,-1 do
    if self.terminal_masks[i] < 0.5 then
      self.tdReturns[i] = 0
    else
      self.tdReturns[i] = self.rewards[i] + self.gamma * self.tdReturns[i+1] * self.terminal_masks[i+1]
    end
  end

  if self.opt.recurrent then self.policyNet_:forget() end
  -- Do the real updating
  for i=1, self.batchIdx do
    -- only do update from non-terminal state
    if self.terminal_masks[i] > 0.5 then
      local action = self.actions[i]
      local V, probability = table.unpack(self.policyNet_:forward(self.states[i]))
      probability:add(TINY_EPSILON) -- could contain 0 -> log(0)= -inf -> theta = nans

      -- For the CI problem, this is a design decision of whether to normalize action distribution during optimization
      -- If normalize it (zero actions that should not be taken at current time step), the potential problem is that
      -- only the relative probability will be adjusted, even though the absolute probability may be adjusted in the wrong
      -- direction. An example is that, for a 'good' action, we may still decrease its absolute probability but increase
      -- the relative probability by decreasing probability of all legal actions. By pwang8. Jan 1, 2018.
      -- Even though it does not have to be a better design.
      local adpT = 0
      if self.opt.env == 'UserSimLearner/CIUserSimEnv' and self.opt.ac_relative_plc then
        -- If it is CI data, pick up actions according to adpType
        if self.states[i][-1][1][-4] > 0.1 then adpT = 1 elseif self.states[i][-1][1][-3] > 0.1 then adpT = 2 elseif self.states[i][-1][1][-2] > 0.1 then adpT = 3 elseif self.states[i][-1][1][-1] > 0.1 then adpT = 4 end
        assert(adpT >=1 and adpT <= 4)
        for i=1, probability:size(1) do
          if i < self.CIActAdpBound[adpT][1] or i > self.CIActAdpBound[adpT][2] then
            probability[i] = 0
          end
        end
        local sumP = probability:sum()
        probability = torch.div(probability, sumP)
      end

      self.vTarget[1] = -2 * self.opt.async_valErr_coef * (self.tdReturns[i] - V)  -- this makes sense, instead of the 0.5 const. It then makes sense if we explain it as result of adopting value loss coefficient, by pwang8

      -- ∇θ logp(s) = 1/p(a) for chosen a, 0 otherwise
      self.policyTarget:zero()
      -- f(s) ∇θ logp(s)
      self.policyTarget[action] = -(self.tdReturns[i] - V) / probability[action] -- Negative target for gradient descent. This calculation should be correct. Same as in pytorch a2c repo. By pwang8.

      -- Calculate (negative of) gradient of entropy of policy (for gradient descent): -(-logp(s) - 1)
      local gradEntropy = torch.log(torch.add(probability, TINY_EPSILON)) + 1

      if self.opt.env == 'UserSimLearner/CIUserSimEnv' and self.opt.ac_relative_plc then
        for i=1, gradEntropy:size(1) do
          if i < self.CIActAdpBound[adpT][1] or i > self.CIActAdpBound[adpT][2] then
            gradEntropy[i] = 0
          end
        end
      end

      -- Add to target to improve exploration (prevent convergence to suboptimal deterministic policy)
      self.policyTarget:add(self.beta, gradEntropy)

      -- Clip gradient if too large
      OptimMisc.clipGradByNorm(self.vTarget, self.opt.rl_grad_clip)
      OptimMisc.clipGradByNorm(self.policyTarget, self.opt.rl_grad_clip)

      self.policyNet_:backward(self.states[i], self.targets)
    else
      if self.opt.recurrent then self.policyNet_:forget() end
    end
  end
end


function A3CAgent:progress(steps)
  self.atomic:inc()
  self.step = self.step + 1
  if self.step % self.progFreq == 0 then
    local progressPercent = 100 * self.step / steps
    local speed = self.progFreq / torch.toc(self.tic)
    self.tic = torch.tic()
    log.info('A3CAgent | step=%d | %.02f%% | speed=%d/sec | η=%.8f',
      self.step, progressPercent, speed, self.optimParams.learningRate)
  end
end


function A3CAgent:probabilisticAction(state)
  local __, probability = table.unpack(self.policyNet_:forward(state))

  if self.opt.env == 'UserSimLearner/CIUserSimEnv' then
    -- If it is CI data, pick up actions according to adpType
    local adpT = 0
    if state[-1][1][-4] > 0.1 then adpT = 1 elseif state[-1][1][-3] > 0.1 then adpT = 2 elseif state[-1][1][-2] > 0.1 then adpT = 3 elseif state[-1][1][-1] > 0.1 then adpT = 4 end
    if adpT==0 then print('====== Get invalid state in A3C ', state[-1][1]) end
    assert(adpT >=1 and adpT <= 4)
    local subAdpActRegion = torch.Tensor(self.CIActAdpBound[adpT][2] - self.CIActAdpBound[adpT][1] + 1):fill(0)
    probability:squeeze()
    for i=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
      subAdpActRegion[i-self.CIActAdpBound[adpT][1]+1] = probability[i]
    end
    -- Have to make sure subAdpActRegion does not sum up to 0 (all 0s) before sent to multinomial()
    subAdpActRegion:add(TINY_EPSILON) -- add a small number to this distribution so it will not sum up to 0
    local _subAdpActSum = subAdpActRegion:sum()
    subAdpActRegion:div(_subAdpActSum)
    local regAct = torch.multinomial(subAdpActRegion, 1):squeeze()
    return self.CIActAdpBound[adpT][1] + regAct - 1
  else
    return torch.multinomial(probability, 1):squeeze()
  end
end

return A3CAgent
