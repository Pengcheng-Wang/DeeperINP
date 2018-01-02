---
--- Created by pwang8.
--- DateTime: 12/31/17 7:22 PM
--- In this script we implemented the PPO actor-critic method.
--- This sciript follows the structure of A3CAgent, and follows the pytorch-a2c-ppo-acktr
--- repo for ppo optimization implementation.
--- PPO means Proximal Policy Optimization.
--- paper: https://arxiv.org/abs/1707.06347
---
local classic = require 'classic'
local optim = require 'optim'
local AsyncAgent = require 'async/AsyncAgent'
require 'modules/sharedRmsProp'

local AsyncPpoAgent,super = classic.class('AsyncPpoAgent', 'AsyncAgent')

local TINY_EPSILON = 1e-6

function AsyncPpoAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
    super._init(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)

    log.info('creating AsyncPpoAgent')

    self.policyNet_ = policyNet:clone()

    self.theta_, self.dTheta_ = self.policyNet_:getParameters()
    self.dTheta_:zero()

    -- this set of targets store gradient of loss over output (df_do)
    self.policyTarget = self.Tensor(self.m)
    self.vTarget = self.Tensor(1)
    self.targets = { self.vTarget, self.policyTarget }

    self.rewards = torch.Tensor(self.batchSize)
    self.actions = torch.ByteTensor(self.batchSize)
    self.states = torch.Tensor(0)
    self.terminal_masks = torch.Tensor(self.batchSize+1):fill(1)  -- this value is 0 if terminal, 1 if not. In s1-a-s2, ter_m[t1] represents whether s1 is terminal
    self.beta = opt.entropyBeta

    self.env:training()

    self.opt = opt
    -- Sorry, adding ugly code here again, just for CI data compatability
    self.CIActAdpBound = {{1, 3}, {4, 5}, {6, 8}, {9, 10}}

    classic.strict(self)
end


function AsyncPpoAgent:learn(steps, from)
    self.step = from or 0
    self.stateBuffer:clear()
    local _trainEpisode = 0

    log.info('AsyncPpoAgent starting | steps=%d', steps)
    local reward, terminal, state = self:start()

    self.states:resize(self.batchSize, table.unpack(state:size():totable()))

    self.tic = torch.tic()
    repeat
        self.theta_:copy(self.theta)
        self.terminal_masks[1] = terminal and 0 or 1    -- this variable indicates whether the state at batchIdx is terminal
        self.batchIdx = 0
        repeat
            self.batchIdx = self.batchIdx + 1
            self.states[self.batchIdx]:copy(state)

            local action = self:probabilisticAction(state)

            self.actions[self.batchIdx] = action

            reward, terminal, state = self:takeAction(action)
            self.rewards[self.batchIdx] = reward
            self.terminal_masks[self.batchIdx+1] = terminal and 0 or 1    -- this variable indicates whether the state at batchIdx is terminal
            -- the indices for rewards and terminal_masks are different. For an s1-a-s2 transition, rewards[t1] represents
            -- the reward got in this transition. But terminal_masks[t1] means whether s1 is a terminal state (0 if it is terminal).

            self:progress(steps)
        until terminal or self.batchIdx == self.batchSize

        _trainEpisode = _trainEpisode + 1   -- counter of training episodes
        self:accumulateGradients(terminal, state)

        self:applyGradients(self.policyNet_, self.dTheta_, self.theta)

        if terminal then
            reward, terminal, state = self:start()
        end
    until self.step >= steps

    log.info('AsyncPpoAgent ended learning steps=%d', steps)
end

-- This accu() function is similar to the NStepQ's implementation, in which td error is calculated using n-step of observation
-- So, it is not very convenient to add an lstm module. If an lstm module is demanded, we can refer to the OneStepQ implementation
function AsyncPpoAgent:accumulateGradients(terminal, state)
    local R = 0
    if not terminal then
        R = self.policyNet_:forward(state)[1]
    end

    for i=self.batchIdx,1,-1 do
        R = self.rewards[i] + self.gamma * R * self.terminal_masks[i+1]

        local action = self.actions[i]
        local V, probability = table.unpack(self.policyNet_:forward(self.states[i]))
        probability:add(TINY_EPSILON) -- could contain 0 -> log(0)= -inf -> theta = nans

        local adpT = 0
        if self.opt.env == 'UserSimLearner/CIUserSimEnv' then
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

        self.vTarget[1] = -0.5 * (R - V)  -- this makes sense, instead of the 0.5 const. It then makes sense if we explain it as result of adopting value loss coefficient, by pwang8

        -- ∇θ logp(s) = 1/p(a) for chosen a, 0 otherwise
        self.policyTarget:zero()
        -- f(s) ∇θ logp(s)
        self.policyTarget[action] = -(R - V) / probability[action] -- Negative target for gradient descent. This calculation should be correct. Same as in pytorch a2c repo. By pwang8.

        -- Calculate (negative of) gradient of entropy of policy (for gradient descent): -(-logp(s) - 1)
        local gradEntropy = torch.log(probability) + 1

        if self.opt.env == 'UserSimLearner/CIUserSimEnv' then
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


function AsyncPpoAgent:progress(steps)
    self.atomic:inc()
    self.step = self.step + 1
    if self.step % self.progFreq == 0 then
        local progressPercent = 100 * self.step / steps
        local speed = self.progFreq / torch.toc(self.tic)
        self.tic = torch.tic()
        log.info('AsyncPpoAgent | step=%d | %.02f%% | speed=%d/sec | η=%.8f',
        self.step, progressPercent, speed, self.optimParams.learningRate)
    end
end


function AsyncPpoAgent:probabilisticAction(state)
    local __, probability = table.unpack(self.policyNet_:forward(state))

    if self.opt.env == 'UserSimLearner/CIUserSimEnv' then
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

return AsyncPpoAgent
