local classic = require 'classic'
local optim = require 'optim'
local QAgent = require 'async/QAgent'
require 'modules/sharedRmsProp'

local OneStepQAgent, super = classic.class('OneStepQAgent', 'QAgent')


function OneStepQAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  self.agentName = 'OneStepQAgent'
  self.lstm = opt.recurrent and self.policyNet:findModules('nn.FastLSTM')[1]
  self.lstmTarget = opt.recurrent and self.targetNet:findModules('nn.FastLSTM')[1]

  self.opt = opt
  -- Sorry, adding ugly code here again, just for CI data compatability
  self.CIActAdpBound = {{1, 3}, {4, 5}, {6, 8}, {9, 10}}

  classic.strict(self)
end


function OneStepQAgent:learn(steps, from)
  self.step = from or 0
  self.policyNet:training()
  self.stateBuffer:clear()
  self.env:training()

  log.info('%s starting | steps=%d | ε=%.2f -> %.2f', self.agentName, steps, self.epsilon, self.epsilonEnd)
  local reward, terminal, state = self:start()

  local action, state_

  self.tic = torch.tic()
  for step1=1,steps do
    if not terminal then
      action = self:eGreedy(state, self.policyNet)
      reward, terminal, state_ = self:takeAction(action)
    else
      reward, terminal, state_ = self:start()
    end

    if state ~= nil then
      self:accumulateGradient(state, action, state_, reward, terminal)
      self.batchIdx = self.batchIdx + 1
    end

    if not terminal then
      state = state_
    else
      if self.lstm then
        self.lstm:forget()
        self.lstmTarget:forget()
      end
      state = nil
    end

    if self.batchIdx == self.batchSize or terminal then
      self:applyGradients(self.policyNet, self.dTheta, self.theta)
      if self.lstm then
        self.lstm:forget()
        self.lstmTarget:forget()
      end
      self.batchIdx = 0
    end

    self:progress(steps)
  end

  log.info('%s ended learning steps=%d ε=%.4f', self.agentName, steps, self.epsilon)
end


function OneStepQAgent:accumulateGradient(state, action, state_, reward, terminal)
  local Y = reward
  if self.lstm then -- LSTM targetNet needs to see all states as well
    self.targetNet:forward(state)
  end
  if not terminal then
      local QPrimes = self.targetNet:forward(state_):squeeze()
      local APrimeMax = QPrimes:max(1):squeeze()    -- So, QPrime is a 1-dim tensor with size 10 (acts)

      -- If it is CI data, pick up actions according to adpType
      local adpT = 0
      if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
          if state_[-1][1][-4] > 0.1 then adpT = 1 elseif state_[-1][1][-3] > 0.1 then adpT = 2 elseif state_[-1][1][-2] > 0.1 then adpT = 3 elseif state_[-1][1][-1] > 0.1 then adpT = 4 end
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
          local dqQPrimes = self.policyNet:forward(state_):squeeze()
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

      Y = Y + self.gamma * APrimeMax
  end

  if self.doubleQ then
    self.QCurr = self.policyNet:forward(state):squeeze()
  end

  local tdErr = Y - self.QCurr[action]

  self:accumulateGradientTdErr(state, action, tdErr, self.policyNet)
end


return OneStepQAgent
