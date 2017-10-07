local classic = require 'classic'
local QAgent = require 'async/AsyncAgent'

local QAgent, super = classic.class('QAgent', 'AsyncAgent')

local EPSILON_ENDS = { 0.1, 0.01, 0.5}
local EPSILON_PROBS = { 0.4, 0.7, 1 }


function QAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  self.super = super

  self.targetNet = targetNet:clone('weight', 'bias')
  self.targetNet:evaluate()

  self.targetTheta = targetTheta
  local __, gradParams = self.policyNet:parameters()
  self.dTheta = nn.Module.flatten(gradParams)
  self.dTheta:zero()

  self.doubleQ = opt.doubleQ

  self.epsilonStart = opt.epsilonStart
  self.epsilon = self.epsilonStart
  self.PALpha = opt.PALpha

  self.target = self.Tensor(self.m)

  self.totalSteps = math.floor(opt.steps / opt.threads)

  self:setEpsilon(opt)
  self.tic = 0
  self.step = 0

  -- Forward state anyway if recurrent
  self.alwaysComputeGreedyQ = opt.recurrent or not self.doubleQ

  self.QCurr = torch.Tensor(0)

  self.opt = opt
  -- Sorry, adding ugly code here again, just for CI data compatability
  self.CIActAdpBound = {{1, 3}, {4, 5}, {6, 8}, {9, 10}}
end


function QAgent:setEpsilon(opt)
  local r = torch.rand(1):squeeze()
  local e = 3
  if r < EPSILON_PROBS[1] then
    e = 1
  elseif r < EPSILON_PROBS[2] then
    e = 2
  end
  self.epsilonEnd = EPSILON_ENDS[e]
  self.epsilonGrad = (self.epsilonEnd - opt.epsilonStart) / opt.epsilonSteps
end


function QAgent:eGreedy(state, net)
  self.epsilon = math.max(self.epsilonStart + (self.step - 1)*self.epsilonGrad, self.epsilonEnd)
  -- When bootstraps is 0 (1 head), size of state is 4*1*25, with 4 the histLen, 1 being # of head, 25 the # of features of state
  -- output for net:forward(state) has 2-dim size 1*10 (10 acts). After squeeze() it is 1-dim of size 10. The 1 should be # of head
  if self.alwaysComputeGreedyQ then
    self.QCurr = net:forward(state):squeeze()
  end

  -- If it is CI data, pick up actions according to adpType
  local adpT = 0
  if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
    if state[-1][1][-4] > 0.1 then adpT = 1 elseif state[-1][1][-3] > 0.1 then adpT = 2 elseif state[-1][1][-2] > 0.1 then adpT = 3 elseif state[-1][1][-1] > 0.1 then adpT = 4 end
    assert(adpT >=1 and adpT <= 4)
  end

  if torch.uniform() < self.epsilon then
    if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
      return torch.random(self.CIActAdpBound[adpT][1], self.CIActAdpBound[adpT][2])
    else
      return torch.random(1,self.m)
    end
  end

  if not self.alwaysComputeGreedyQ then
    self.QCurr = net:forward(state):squeeze()
  end

  if self.opt.env == 'UserSimLearner/CIUserSimEnv' then   -- Todo: pwang8. Check correctness
    local maxAct = self.CIActAdpBound[adpT][1]
    local maxActQValue = self.QCurr[maxAct]
    for i=maxAct+1, self.CIActAdpBound[adpT][2] do
      if self.QCurr[i] > maxActQValue then
        maxActQValue = self.QCurr[i]
        maxAct = i
      end
    end
    return maxAct
  else
    local _, maxIdx = self.QCurr:max(1) --Q:max(1)
    return maxIdx[1]
  end
end


function QAgent:progress(steps)
  self.step = self.step + 1
  if self.atomic:inc() % self.tau == 0 then
    self.targetTheta:copy(self.theta)
    if self.tau>1000 then
      log.info('QAgent | updated targetNetwork at %d', self.atomic:get()) 
    end
  end
  if self.step % self.progFreq == 0 then
    local progressPercent = 100 * self.step / steps
    local speed = self.progFreq / torch.toc(self.tic)
    self.tic = torch.tic()
    log.info('AsyncAgent | step=%d | %.02f%% | speed=%d/sec | ε=%.2f -> %.2f | η=%.8f',
      self.step, progressPercent, speed ,self.epsilon, self.epsilonEnd, self.optimParams.learningRate)
  end
end


function QAgent:accumulateGradientTdErr(state, action, tdErr, net)
  if self.tdClip > 0 then
      if tdErr > self.tdClip then tdErr = self.tdClip end
      if tdErr <-self.tdClip then tdErr =-self.tdClip end
  end

  self.target:zero()
  self.target[action] = -tdErr

  net:backward(state, self.target)
end


return QAgent

