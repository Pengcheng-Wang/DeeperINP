local classic = require 'classic'
local optim = require 'optim'
local AsyncAgent = require 'async/AsyncAgent'
require 'modules/sharedRmsProp'

local A3CAgent,super = classic.class('A3CAgent', 'AsyncAgent')

local TINY_EPSILON = 1e-20

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

  log.info('A3CAgent starting | steps=%d', steps)
  local reward, terminal, state = self:start()

  self.states:resize(self.batchSize, table.unpack(state:size():totable()))

  self.tic = torch.tic()
  repeat
    self.theta_:copy(self.theta)
    self.batchIdx = 0
    repeat
      self.batchIdx = self.batchIdx + 1
      self.states[self.batchIdx]:copy(state)

--      local V, probability = table.unpack(self.policyNet_:forward(state)) -- For CI sim, V is a 1-dim, size-1 tensor, probability is a 1-dim, size-10 (act#) tensor.
--      local action = torch.multinomial(probability, 1):squeeze()
      local action = self:probabilisticAction(state)  -- Todo: pwang8. Check correctness

      self.actions[self.batchIdx] = action

      reward, terminal, state = self:takeAction(action)
      self.rewards[self.batchIdx] = reward

      self:progress(steps)
    until terminal or self.batchIdx == self.batchSize

    self:accumulateGradients(terminal, state)

    if terminal then
      reward, terminal, state = self:start()
    end

    self:applyGradients(self.policyNet_, self.dTheta_, self.theta)
  until self.step >= steps

  log.info('A3CAgent ended learning steps=%d', steps)
end

-- todo:pwang8. Take a careful look at this implmentation. Check it against the baseline a2c code. Dec 29, 2017. Morning :D
function A3CAgent:accumulateGradients(terminal, state)
  local R = 0
  if not terminal then
    R = self.policyNet_:forward(state)[1]
  end

  for i=self.batchIdx,1,-1 do
    R = self.rewards[i] + self.gamma * R

    local action = self.actions[i]
    local V, probability = table.unpack(self.policyNet_:forward(self.states[i]))
    probability:add(TINY_EPSILON) -- could contain 0 -> log(0)= -inf -> theta = nans

    local adpT = 0
    if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
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
      probability:add(TINY_EPSILON)
    end

    self.vTarget[1] = -0.5 * (R - V)

    -- ∇θ logp(s) = 1/p(a) for chosen a, 0 otherwise
    self.policyTarget:zero()
    -- f(s) ∇θ logp(s)
    self.policyTarget[action] = -(R - V) / probability[action] -- Negative target for gradient descent

    -- Calculate (negative of) gradient of entropy of policy (for gradient descent): -(-logp(s) - 1)
    local gradEntropy = torch.log(probability) + 1

    if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
      for i=1, gradEntropy:size(1) do
        if i < self.CIActAdpBound[adpT][1] or i > self.CIActAdpBound[adpT][2] then
          gradEntropy[i] = 0
        end
      end
    end

    -- Add to target to improve exploration (prevent convergence to suboptimal deterministic policy)
    self.policyTarget:add(self.beta, gradEntropy)
    
    self.policyNet_:backward(self.states[i], self.targets)
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


function A3CAgent:probabilisticAction(state)  -- Todo: pwang8. Check correctness. This is borrowed from ValidationAgent
  local __, probability = table.unpack(self.policyNet_:forward(state))

  if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
    -- If it is CI data, pick up actions according to adpType
    local adpT = 0
    if state[-1][1][-4] > 0.1 then adpT = 1 elseif state[-1][1][-3] > 0.1 then adpT = 2 elseif state[-1][1][-2] > 0.1 then adpT = 3 elseif state[-1][1][-1] > 0.1 then adpT = 4 end
    assert(adpT >=1 and adpT <= 4)
    local subAdpActRegion = torch.Tensor(self.CIActAdpBound[adpT][2] - self.CIActAdpBound[adpT][1] + 1):fill(0)
    probability:squeeze()
    for i=self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2] do
      subAdpActRegion[i-self.CIActAdpBound[adpT][1]+1] = probability[i]
    end
    -- Have to make sure subAdpActRegion does not sum up to 0 (all 0s) before sent to multinomial()
    subAdpActRegion:add(TINY_EPSILON) -- add a small number to this distribution so it will not sum up to 0
    local regAct = torch.multinomial(subAdpActRegion, 1):squeeze()
    return self.CIActAdpBound[adpT][1] + regAct - 1
  else
    return torch.multinomial(probability, 1):squeeze()
  end
end

return A3CAgent
