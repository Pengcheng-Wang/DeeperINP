--
-- User: pwang8
-- Date: 1/31/17
-- Time: 9:32 PM
-- This is the main script for the DQN program using CI user simulation model
--

local Setup = require 'Setup'
local Master = require 'Master'
local AsyncMaster = require 'async/AsyncMaster'
local AsyncEvaluation = require 'async/AsyncEvaluation'

-- Parse options and perform setup
local setup = Setup(arg)
local opt = setup.opt

-- Start master experiment runner
if opt.async then
    if opt.mode == 'train' then
        local master = AsyncMaster(opt)
        master:start()
    elseif opt.mode == 'eval' then
        local eval = AsyncEvaluation(opt)
        eval:evaluate()
    elseif opt.mode == 'is' then
        local eval = AsyncEvaluation(opt)
        eval:ISevaluate()
    end
else
    local master = Master(opt)

    if opt.mode == 'train' then
        master:train()
    elseif opt.mode == 'eval' then
        master:evaluate()
    elseif opt.mode == 'is' then
        master:ISevaluate()
    end
end


--
--local obv, score, term, adpType
--term = false
--obv, adpType = CIUserBehaviorGen:start()
--print('^### Outside in main\n state:', obv, '\n type:', adpType)
--while not term do
--    local rndAdpAct = torch.random(fr.ciAdpActRanges[adpType][1], fr.ciAdpActRanges[adpType][2])
--    print('^--- Adaptation type', adpType, 'Random act choice: ', rndAdpAct)
--    score, obv, term, adpType = CIUserBehaviorGen:step(rndAdpAct)
--    print('^### Outside in main\n state:', obv, '\n type:', adpType, '\n score:', score, ',term:', term)
--end
