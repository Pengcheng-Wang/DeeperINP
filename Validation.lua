local _ = require 'moses'
local classic = require 'classic'
local Evaluator = require 'Evaluator'
local TableSet = require 'MyMisc.TableSetMisc'

local Validation = classic.class('Validation')

function Validation:_init(opt, agent, env, display)
  self.opt = opt
  self.agent = agent
  self.env = env

  self.hasDisplay = false
  if display then
    self.hasDisplay = true
    self.display = display
  end

  -- Create (Atari normalised score) evaluator
  self.evaluator = Evaluator(opt.game)

  self.bestValScore = _.max(self.agent.valScores) or -math.huge -- Retrieve best validation score from agent if available

  classic.strict(self)
end


function Validation:validate()
  log.info('Validating')
  -- Set environment and agent to evaluation mode
  self.env:evaluate() -- It's a little confusing, since in rlenvs, only Atari has def of evaluate(). But no errors thrown when Catch is run
  self.agent:evaluate()

  -- Start new game
  local reward, terminal = 0, false
  local state, adpType = self.env:start()    -- Todo: pwang8. This has been changed a little for compatibility with CI sim

  -- Validation variables
  local valEpisode = 1
  local valEpisodeScore = 0
  local valTotalScore = 0
  local valStepStrFormat = '%0' .. (math.floor(math.log10(self.opt.valSteps)) + 1) .. 'd' -- String format for padding step with zeros

  for valStep = 1, self.opt.valSteps do
    -- Observe and choose next action (index)
    local action = self.agent:observe(reward, state, terminal)
    if not terminal then
      -- Act on environment
      reward, state, terminal, adpType = self.env:step(action)
      -- Track score
      valEpisodeScore = valEpisodeScore + reward
    else
      -- Print score every 10 episodes
      if valEpisode % 10 == 0 then
        log.info('[VAL] Steps: ' .. string.format(valStepStrFormat, valStep) .. '/' .. self.opt.valSteps .. ' | Episode ' .. valEpisode .. ' | Score: ' .. valEpisodeScore)
      end

      -- Start a new episode
      valEpisode = valEpisode + 1
      reward, terminal = 0, false
      state, adpType = self.env:start()    -- Todo: pwang8. This has been changed a little for compatibility with CI sim
      valTotalScore = valTotalScore + valEpisodeScore -- Only add to total score at end of episode
      valEpisodeScore = reward -- Reset episode score
    end

    -- Display (if available)
    if self.hasDisplay then
      self.display:display(self.agent, self.env:getDisplay())
    end
  end

  -- If no episodes completed then use score from incomplete episode
  if valEpisode == 1 then
    valTotalScore = valEpisodeScore
  end

  -- Print total and average score
  log.info('Total Score: ' .. valTotalScore)
  valTotalScore = valTotalScore/math.max(valEpisode - 1, 1) -- Only average score for completed episodes in general
  log.info('Average Score: ' .. valTotalScore)
  -- Pass to agent (for storage and plotting)
  self.agent.valScores[#self.agent.valScores + 1] = valTotalScore
  -- Calculate normalised score (if valid)
  local normScore = self.evaluator:normaliseScore(valTotalScore)
  if normScore then
    log.info('Normalised Score: ' .. normScore)
    self.agent.normScores[#self.agent.normScores + 1] = normScore
  end

  -- Visualise convolutional filters
  self.agent:visualiseFilters()

  -- Use transitions sampled for validation to test performance
  local avgV, avgTdErr = self.agent:validate()
  log.info('Average V: ' .. avgV)
  log.info('Average δ: ' .. avgTdErr)
  
  -- Save latest weights
  log.info('Saving weights')
  self.agent:saveWeights(paths.concat(self.opt.experiments, self.opt._id, 'last.weights.t7'))

  -- Save "best weights" if best score achieved
  if valTotalScore > self.bestValScore then
    log.info('New best average score')
    self.bestValScore = valTotalScore

    log.info('Saving new best weights')
    self.agent:saveWeights(paths.concat(self.opt.experiments, self.opt._id, 'best.weights.t7'))
  end

  if (self.agent.globals.step/self.opt.progFreq) % 10 == 0 then
    log.info('Saving weights on step' .. self.agent.globals.step/self.opt.progFreq)
    self.agent:saveWeights(paths.concat(self.opt.experiments, self.opt._id,
      'Epoch_' .. self.agent.globals.step/self.opt.progFreq .. string.format('%.2f', valTotalScore) .. '.weights.t7'))
  end
  
  -- Set environment and agent to training mode
  self.env:training()
  self.agent:training()
end


function Validation:evaluate()
  log.info('Evaluation mode')
  -- Set environment and agent to evaluation mode
  self.env:evaluate()
  self.agent:evaluate()

  -- Start new game
  local reward, terminal = 0, false
  local state, adpType = self.env:start()    -- Todo: pwang8. This has been changed a little for compatibility with CI sim

  -- Validation variables
  local valEpisode = 1
  local valEpisodeScore = 0
  local valTotalScore = 0
  local valStep = 1
  local valStepStrFormat = '%0' .. (math.floor(math.log10(self.opt.valSteps)) + 1) .. 'd' -- String format for padding step with zeros

  while valEpisode <= self.opt.evaTrajs do
    -- Observe and choose next action (index)
    local action = self.agent:observe(reward, state, terminal)
    if not terminal then
      -- Act on environment
      reward, state, terminal, adpType = self.env:step(action)
      -- Track score
      valEpisodeScore = valEpisodeScore + reward
    else
      -- Print score every 10 episodes
      if valEpisode % 10 == 0 then
        local avgScore = valTotalScore/math.max(valEpisode - 1, 1)
        log.info('[VAL] Steps: ' .. valStep .. ' | Episode ' .. valEpisode .. ' | Score: ' .. valEpisodeScore
            .. ' | TotScore: ' .. valTotalScore .. ' | AvgScore: %.2f', avgScore)
      end

      -- Start a new episode
      valEpisode = valEpisode + 1
      reward, terminal = 0, false
      state, adpType = self.env:start()    -- Todo: pwang8. This has been changed a little for compatibility with CI sim
      valTotalScore = valTotalScore + valEpisodeScore -- Only add to total score at end of episode
      valEpisodeScore = reward -- Reset episode score
    end
    valStep = valStep + 1
  end

  log.info('[VAL] Final evaluation avg score: ' .. valTotalScore/self.opt.evaTrajs)

  -- Record (if available)
  if self.hasDisplay then
    self.display:createVideo()
  end
end


function Validation:ISevaluate()
  log.info('IS Evaluation mode')
  -- Set environment and agent to evaluation mode
  self.env:evaluate()
  self.agent:evaluate()

  local userSim = self.env.CIUSim

  local totalScoreIs = 0
  local sumOfProbWeights = 0
  local totalScoreIsDiscout = 0
  for uid, uRec in pairs(userSim.realUserRLTerms) do
    local rwd = userSim.realUserRLRewards[uid][1]
    local weight = 1.0
    local weightDiscount = 1.0
    for k, v in pairs(uRec) do
      if v < 1 then -- not terminal
        local _, actDist = self.agent:observe(0, userSim.realUserRLStatePrepInd[uid][k], false)
        local randprob = 0.333333
        if userSim.realUserRLTypes[uid][k] == userSim.CIFr.ciAdp_BryceSymp or
                userSim.realUserRLTypes[uid][k] == userSim.CIFr.ciAdp_PresentQuiz then
          randprob = 0.5
        end
        weight = weight * (actDist[userSim.realUserRLActs[uid][k]] / randprob)
      else  -- terminal
        self.agent:observe(rwd, userSim.realUserRLStatePrepInd[uid][k], true) -- this is necessary is lstm is used, bcz forget() will be called in observe()
        sumOfProbWeights = sumOfProbWeights + weight  -- This is the sum of all trajectory appearance probabilities (no negative values)
        weight = weight * rwd -- rwd can be -1 or 1
        weightDiscount = weight * math.pow(self.opt.gamma, k-1)
      end
    end
    if self.opt.isevaprt then log.info('IS policy weigthed value:' .. weight .. ', discnt: ' .. weightDiscount) end
    totalScoreIs = totalScoreIs + weight
    totalScoreIsDiscout = totalScoreIsDiscout + weightDiscount
  end

  local trjCnt = TableSet.countsInSet(userSim.realUserRLTerms)
  log.info('Importance Sampling rewards on test set: ' .. totalScoreIs/sumOfProbWeights .. ', total: ' .. totalScoreIs ..
          'Discount Importance Sampling rewards on test set: '.. totalScoreIsDiscout/sumOfProbWeights .. ', total: ' .. totalScoreIsDiscout)

end


return Validation
