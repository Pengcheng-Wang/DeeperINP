local classic = require 'classic'
local signal = require 'posix.signal'
local Singleton = require 'structures/Singleton'
local Agent = require 'Agent'
local Display = require 'Display'
local Validation = require 'Validation'

local Master = classic.class('Master')

-- Sets up environment and agent
function Master:_init(opt)
  self.opt = opt
  self.verbose = opt.verbose
  self.learnStart = opt.learnStart  -- For the Catch environment, the learnStart is set to 50000
  self.progFreq = opt.progFreq  -- Reporting progress (printing) frequency
  self.reportWeights = opt.reportWeights
  self.noValidation = opt.noValidation  -- Disable asynchronous validation, false as default
  self.valFreq = opt.valFreq  -- In the Catch env, this value is set to 250000. #epochs = steps/valFreq
  self.experiments = opt.experiments  -- This indicates the directory where log files are stored
  self._id = opt._id

  -- Set up singleton global object for transferring step
  self.globals = Singleton({step = 1}) -- Initial step

  -- Initialise environment
  log.info('Setting up ' .. opt.env)  -- name of the enviornment, like rlenvs.Catch
  local Env = require(opt.env)
  self.env = Env(opt) -- Environment instantiation

  -- Create DQN agent
  log.info('Creating DQN')
  self.agent = Agent(opt, self.env) -- Here, I made the modification to use self.env as a parameter just for the usage of applying raw data directly in training. on Mar 22, 2017
  if paths.filep(opt.network) then
    -- Load saved agent if specified
    log.info('Loading pretrained network weights')
    self.agent:loadWeights(opt.network)
  elseif paths.filep(paths.concat(opt.experiments, opt._id, 'agent.t7')) then
    -- Ask to load saved agent if found in experiment folder (resuming training)
    log.info('Saved agent found - load (y/n)?')
    if io.read() == 'y' then
      log.info('Loading saved agent')
      self.agent = torch.load(paths.concat(opt.experiments, opt._id, 'agent.t7'))

      -- Reset globals (step) from agent
      Singleton.setInstance(self.agent.globals)
      self.globals = Singleton.getInstance()

      -- Switch saliency style
      self.agent:setSaliency(opt.saliency)
    end
  end

  -- Start gaming
  log.info('Starting ' .. opt.env)
  if opt.game ~= '' then
    log.info('Starting game: ' .. opt.game)
  end
  local state, adpType = self.env:start()

  -- Set up display (if available)
  self.hasDisplay = false
  if opt.displaySpec then
    self.hasDisplay = true
    self.display = Display(opt, self.env:getDisplay())
  end

  -- Set up validation (with display if available)
  self.validation = Validation(opt, self.agent, self.env, self.display)

  -- Print out number of trainable parameters in NN
  print("Number of trainable params in QN: ", self.agent.theta:nElement())

  classic.strict(self)
end

-- Trains agent
function Master:train()
  log.info('Training mode')

  -- Catch CTRL-C to save
  self:catchSigInt()

  local reward, terminal = 0, false
  local state, adpType = self.env:start()   -- Todo: pwang8. This has been changed a little for compatibility with CI sim

  -- Set environment and agent to training mode
  self.env:training()
  self.agent:training()

  -- Training variables (reported in verbose mode)
  local episode = 1
  local episodeScore = reward

  -- Training loop
  local initStep = self.globals.step -- Extract step
  local stepStrFormat = '%0' .. (math.floor(math.log10(self.opt.steps)) + 1) .. 'd' -- String format for padding step with zeros
  for step = initStep, self.opt.steps do
    self.globals.step = step -- Pass step number to globals for use in other modules

    -- Observe results of previous transition (r, s', terminal') and choose next action (index)
    local action = self.agent:observe(reward, state, terminal) -- As results received, learn in training mode
    if not terminal then
      -- Act on environment (to cause transition)
--      if adpType == 1 then action = 1 elseif adpType == 2 then action = 5 elseif adpType == 3 then action = 6 else action = 9 end   -- Todo:pwang8. Delete this
      reward, state, terminal, adpType = self.env:step(action)
      -- Track score
      episodeScore = episodeScore + reward
    else
      if self.verbose then
        -- Print score for episode
        log.info('Steps: ' .. string.format(stepStrFormat, step) .. '/' .. self.opt.steps .. ' | Episode ' .. episode .. ' | Score: ' .. episodeScore)
      end

      -- Start a new episode
      episode = episode + 1
      reward, terminal = 0, false
      state, adpType = self.env:start()    -- Todo: pwang8. This has been changed a little for compatibility with CI sim
      episodeScore = reward -- Reset episode score
    end

    -- Display (if available)
    if self.hasDisplay then
      self.display:display(self.agent, self.env:getDisplay())
    end

    -- Trigger learning after a while (wait to accumulate experience)
    if step == self.learnStart then
      log.info('Learning started')
    end

    -- Report progress
    if step % self.progFreq == 0 then
      log.info('Steps: ' .. string.format(stepStrFormat, step) .. '/' .. self.opt.steps)
      -- Report weight and weight gradient statistics
      if self.reportWeights then
        local reports = self.agent:report()
        for r = 1, #reports do
          log.info(reports[r])
        end
      end
    end

    -- Validate. Should set this valFreq to a relatively large number. Otherwise gnuplot may have problem
    if not self.noValidation and step >= self.learnStart and step % self.valFreq == 0 then
      self.validation:validate() -- Sets env and agent to evaluation mode and then back to training mode

      log.info('Resuming training')
      -- Start new game (as previous one was interrupted)
      reward, terminal = 0, false
      state, adpType = self.env:start()    -- Todo: pwang8. This has been changed a little for compatibility with CI sim
      episodeScore = reward
    end
  end

  log.info('Finished training')
end

function Master:evaluate()
  self.validation:evaluate() -- Sets env and agent to evaluation mode
end

function Master:ISevaluate()
    self.validation:ISevaluate() -- Sets env and agent to evaluation mode
end

-- Sets up SIGINT (Ctrl+C) handler to save network before quitting
function Master:catchSigInt()
  signal.signal(signal.SIGINT, function(signum)
    log.warn('SIGINT received')
    log.info('Save agent (y/n)?')
    if io.read() == 'y' then
      log.info('Saving agent')
      torch.save(paths.concat(self.experiments, self._id, 'agent.t7'), self.agent) -- Save agent to resume training
    end
    log.warn('Exiting')
    os.exit(128 + signum)
  end)
end

return Master
