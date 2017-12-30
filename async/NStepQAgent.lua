local classic = require 'classic'
local optim = require 'optim'
local QAgent = require 'async/QAgent'
require 'modules/sharedRmsProp'

local NStepQAgent, super = classic.class('NStepQAgent', 'QAgent')


function NStepQAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  self.policyNet_ = self.policyNet:clone()
  self.policyNet_:training()
  self.theta_, self.dTheta_ = self.policyNet_:getParameters()
  self.dTheta_:zero()

  self.rewards = torch.Tensor(self.batchSize)
  self.actions = torch.ByteTensor(self.batchSize)
  self.states = torch.Tensor(0)

  self.env:training()

  self.alwaysComputeGreedyQ = false

  self.opt = opt
  -- Sorry, adding ugly code here again, just for CI data compatability
  self.CIActAdpBound = {{1, 3}, {4, 5}, {6, 8}, {9, 10}}

  classic.strict(self)
end


function NStepQAgent:learn(steps, from)
  self.step = from or 0
  self.stateBuffer:clear()

  log.info('NStepQAgent starting | steps=%d | ε=%.2f -> %.2f', steps, self.epsilon, self.epsilonEnd)
  local reward, terminal, state = self:start()

  self.states:resize(self.batchSize, table.unpack(state:size():totable()))
  self.tic = torch.tic()
  repeat
    self.theta_:copy(self.theta)
    self.batchIdx = 0
    repeat
      self.batchIdx = self.batchIdx + 1
      self.states[self.batchIdx]:copy(state)

      local action = self:eGreedy(state, self.policyNet_)
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

  log.info('NStepQAgent ended learning steps=%d ε=%.4f', steps, self.epsilon)
end


function NStepQAgent:accumulateGradients(terminal, state)
  local R = 0
  if not terminal then
    local QPrimes = self.targetNet:forward(state):squeeze()   -- It seems list NStepQ does not support bootstrapping right now.
    local APrimeMax = QPrimes:squeeze():max(1)    -- So, QPrime is a 1-dim tensor with size 10 (acts)

    -- If it is CI data, pick up actions according to adpType
    local adpT = 0
    if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
      if state[-1][1][-4] > 0.1 then adpT = 1 elseif state[-1][1][-3] > 0.1 then adpT = 2 elseif state[-1][1][-2] > 0.1 then adpT = 3 elseif state[-1][1][-1] > 0.1 then adpT = 4 end
      assert(adpT >=1 and adpT <= 4)
    end

    if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
      local maxAct = self.CIActAdpBound[adpT][1]
      local maxActQValue = QPrimes[maxAct]
      for i=maxAct+1, self.CIActAdpBound[adpT][2] do
        if QPrimes[i] > maxActQValue then
          maxActQValue = QPrimes[i]
          maxAct = i
        end
      end
      APrimeMax = maxActQValue
    end

    if self.doubleQ then
        local dqQPrimes = self.policyNet_:forward(state):squeeze()
        local _,APrimeMaxInds = dqQPrimes:max(1)

        if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
          local maxAct = self.CIActAdpBound[adpT][1]
          local maxActQValue = dqQPrimes[maxAct]
          for i=maxAct+1, self.CIActAdpBound[adpT][2] do
            if dqQPrimes[i] > maxActQValue then
              maxActQValue = dqQPrimes[i]
              maxAct = i
            end
          end
          APrimeMaxInds[1] = maxAct
        end

        APrimeMax = QPrimes[APrimeMaxInds[1]]
    end
    R = APrimeMax
  end

  for i=self.batchIdx,1,-1 do
    R = self.rewards[i] + self.gamma * R
    local Q_i = self.policyNet_:forward(self.states[i]):squeeze()
    local tdErr = R - Q_i[self.actions[i]]
    print('###### i:', i)
    print('Type R:', type(R))
    print('Type Q_i:', type(Q_i[self.actions[i]]))
    print(tdErr)
    print('Type tdErr:', type(tdErr))
    self:accumulateGradientTdErr(self.states[i], self.actions[i], tdErr, self.policyNet_)
  end
end


return NStepQAgent
