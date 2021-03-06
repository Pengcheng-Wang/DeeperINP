require 'logroll'
local _ = require 'moses'
local classic = require 'classic'
local cjson = require 'cjson'

local Setup = classic.class('Setup')

-- Performs global setup
function Setup:_init(arg)
  -- Create log10 for Lua 5.2
  if not math.log10 then
    math.log10 = function(x)
      return math.log(x, 10)
    end
  end

  -- Parse command-line options
  self.opt = self:parseOptions(arg)

  -- Create experiment directory
  if not paths.dirp(self.opt.experiments) then
    paths.mkdir(self.opt.experiments)
  end
  paths.mkdir(paths.concat(self.opt.experiments, self.opt._id))
  -- Save options for reference
  local file = torch.DiskFile(paths.concat(self.opt.experiments, self.opt._id, 'opts.json'), 'w')
  file:writeString(cjson.encode(self.opt))
  file:close()

  -- Set up logging
  local flog = logroll.file_logger(paths.concat(self.opt.experiments, self.opt._id, 'log.txt'))
  local plog = logroll.print_logger()
  log = logroll.combine(flog, plog) -- Global logger

  -- Validate command-line options (logging errors)
  self:validateOptions()

  -- Augment environments to meet spec
  self:augmentEnv()   -- This is a smart design

  -- Torch setup
  log.info('Setting up Torch7')
  -- Set number of BLAS threads
  torch.setnumthreads(self.opt.threads)
  -- Set default Tensor type (float is more efficient than double)
  torch.setdefaulttensortype(self.opt.tensorType)
  -- Set manual seed
  torch.manualSeed(self.opt.seed)

  -- Tensor creation function for removing need to cast to CUDA if GPU is enabled
  -- TODO: Replace with local functions across codebase
  self.opt.Tensor = function(...)
    return torch.Tensor(...)
  end

  -- GPU setup
  if self.opt.gpu > 0 then
    log.info('Setting up GPU')
    cutorch.setDevice(self.opt.gpu)
    -- Set manual seeds using random numbers to reduce correlations
    cutorch.manualSeed(torch.random())
    -- Replace tensor creation function
    self.opt.Tensor = function(...)
      return torch.CudaTensor(...)
    end
  end

  classic.strict(self)
end

-- Parses command-line options
function Setup:parseOptions(arg)
  -- Detect and use GPU 1 by default
  local cuda = pcall(require, 'cutorch')

  local cmd = torch.CmdLine()
  -- Base Torch7 options
  cmd:option('-seed', 1, 'Random seed')
  cmd:option('-threads', 4, 'Number of BLAS or async threads')
  cmd:option('-tensorType', 'torch.FloatTensor', 'Default tensor type')
  cmd:option('-gpu', cuda and 1 or 0, 'GPU device ID (0 to disable)')
  cmd:option('-cudnn', 'false', 'Utilise cuDNN (if available)')
  -- Environment options
  cmd:option('-env', 'UserSimLearner/CIUserSimEnv', 'Environment class (Lua file to be loaded/rlenv)')
  cmd:option('-zoom', 1, 'Display zoom (requires QT)')
  cmd:option('-game', '', 'Name of Atari ROM (stored in "roms" directory)')
  -- Training vs. evaluate mode
  cmd:option('-mode', 'train', 'Train vs. test mode: train|eval|is')
  -- State preprocessing options (for visual states)
  cmd:option('-height', 0, 'Resized screen height (0 to disable)')
  cmd:option('-width', 0, 'Resize screen width (0 to disable)')
  cmd:option('-colorSpace', '', 'Colour space conversion (screen is RGB): <none>|y|lab|yuv|hsl|hsv|nrgb')
  -- Model options
  cmd:option('-modelBody', 'models.CISim', 'Path to Torch nn model to be used as DQN "body"')
  cmd:option('-ciTemCnn', 2, 'If models.CISim is used, this is the number of temporal cnn layers added in the DRL model')
  cmd:option('-drlCnnKernelWidth', 1, 'Kernel width of temporal cnn module in DRL network')
  cmd:option('-drlCnnConnType', 'v6', 'Temporal cnn residual connection type in DRL network, only can apply v5-v8')
  cmd:option('-hiddenSize', 512, 'Number of units in the hidden fully connected layer')
  cmd:option('-histLen', 4, 'Number of consecutive states processed/used for backpropagation-through-time') -- DQN standard is 4, DRQN is 10
  cmd:option('-duel', 'true', 'Use dueling network architecture (learns advantage function)')
  cmd:option('-rlnnLinear', 'false', 'Use linear model for rl value function approximation')
  cmd:option('-bootstraps', 10, 'Number of bootstrap heads (0 to disable)')
  --cmd:option('-bootstrapMask', 1, 'Independent probability of masking a transition for each bootstrap head ~ Ber(bootstrapMask) (1 to disable)')
  cmd:option('-recurrent', 'false', 'Use recurrent connections')
  -- Experience replay options
  cmd:option('-discretiseMem', 'false', 'Discretise states to integers ∈ [0, 255] for storage')
  cmd:option('-memSize', 1e6, 'Experience replay memory size (number of tuples)')
  cmd:option('-memSampleFreq', 4, 'Interval of steps between sampling from memory to learn')
  cmd:option('-memNSamples', 1, 'Number of times to sample per learning step')
  cmd:option('-memPriority', '', 'Type of prioritised experience replay: <none>|rank|proportional') -- TODO: Implement proportional prioritised experience replay
  cmd:option('-alpha', 0.65, 'Prioritised experience replay exponent α') -- Best vals are rank = 0.7, proportional = 0.6
  cmd:option('-betaZero', 0.45, 'Initial value of importance-sampling exponent β') -- Best vals are rank = 0.5, proportional = 0.4
  -- Reinforcement learning parameters
  cmd:option('-gamma', 0.99, 'Discount rate γ')
  cmd:option('-epsilonStart', 1, 'Initial value of greediness ε')
  cmd:option('-epsilonEnd', 0.01, 'Final value of greediness ε') -- Tuned DDQN final greediness (1/10 that of DQN)
  cmd:option('-epsilonSteps', 1e6, 'Number of steps to linearly decay epsilonStart to epsilonEnd') -- Usually same as memory size
  cmd:option('-tau', 30000, 'Steps between target net updates τ') -- Tuned DDQN target net update interval (3x that of DQN)
  cmd:option('-rewardClip', 1, 'Clips reward magnitude at rewardClip (0 to disable)')
  cmd:option('-tdClip', 1, 'Clips TD-error δ magnitude at tdClip (0 to disable)')
  cmd:option('-doubleQ', 'true', 'Use Double Q-learning')
  -- Note from Georg Ostrovski: The advantage operators and Double DQN are not entirely orthogonal as the increased action gap seems to reduce the statistical bias that leads to value over-estimation in a similar way that Double DQN does
  cmd:option('-PALpha', 0.9, 'Persistent advantage learning parameter α (0 to disable)')
  -- Training options
  cmd:option('-optimiser', 'rmspropm', 'Training algorithm') -- RMSProp with momentum as found in "Generating Sequences With Recurrent Neural Networks"
  cmd:option('-eta', 0.0000625, 'Learning rate η') -- Prioritied experience replay learning rate (1/4 that of DQN; does not account for Duel as well)
  cmd:option('-momentum', 0.95, 'Gradient descent momentum')
  cmd:option('-batchSize', 32, 'Minibatch size')
  cmd:option('-steps', 5e7, 'Training iterations (steps)') -- Frame := step in ALE; Time step := consecutive frames treated atomically by the agent
  cmd:option('-learnStart', 50000, 'Number of steps after which learning starts')
  cmd:option('-gradClip', 10, 'Clips L2 norm of gradients at gradClip (0 to disable)')
  cmd:option('-rlDropout', 0.1, 'Dropout value for DRL network')
  -- Evaluation options
  cmd:option('-progFreq', 10000, 'Interval of steps between reporting progress')
  cmd:option('-reportWeights', 'false', 'Report weight and weight gradient statistics')
  cmd:option('-noValidation', 'false', 'Disable asynchronous agent validation thread') -- TODO: Make behaviour consistent across Master/AsyncMaster
  cmd:option('-valFreq', 250000, 'Interval of steps between validating agent') -- valFreq steps is used as an epoch, hence #epochs = steps/valFreq
  cmd:option('-valSteps', 125000, 'Number of steps to use for validation')
  cmd:option('-valSize', 500, 'Number of transitions to use for calculating validation statistics')
  cmd:option('-evaTrajs', 500, 'Number of trajectories to use for evaluation')
  cmd:option('-isevaprt', 'false', 'Whether to print importance sampling based policy value for each test trajectory')
  cmd:option('-ac_greedy', 'false', 'Whether actor-critic agent should pick greedy action in evaluation')
  -- Async options
  cmd:option('-async', '', 'Async agent: <none>|Sarsa|OneStepQ|NStepQ|A3C|PPO')   -- newly adding PPO async agent
  cmd:option('-actor_critic', '', 'Whether the DRL model is actor-critic. This opt item should not be set by user in command')
  cmd:option('-ac_relative_plc', 'false', 'Whether to utilize relative(non-absolute policy output) in param optimization in actor-critic models for CI')
  cmd:option('-rmsEpsilon', 0.1, 'Epsilon for sharedRmsProp')
  cmd:option('-entropyBeta', 0.01, 'Policy entropy regularisation β')
  cmd:option('-asyncOptimFreq', 1, 'Param updating frequency of async RL models. This is the number of interaction sequences')
  cmd:option('-asyncRecrRho', 4, 'The rho (maximum BPTT steps) param for FastLSTM module in async RL models')
  cmd:option('-ppo_optim_epo', 4, 'Optimization epoch number using each batch of data for PPO agent')
  cmd:option('-ppo_clip_thr', 0.2, 'Threshold parameter for advantage-based error clipping in PPO')
  cmd:option('-rl_grad_clip', 1, 'The maximum grad norm allowed in optimization for async RL models (not used by DQN-based models because they already got tdErr clip)')
  cmd:option('-async_valErr_coef', 0.25, 'The coefficient of value estimation error in actor-critic models')
  -- ALEWrap options
  cmd:option('-fullActions', 'false', 'Use full set of 18 actions')
  cmd:option('-actRep', 4, 'Times to repeat action') -- Independent of history length
  cmd:option('-randomStarts', 30, 'Max number of no-op actions played before presenting the start of each training episode')
  cmd:option('-poolFrmsType', 'max', 'Type of pooling over previous emulator frames: max|mean')
  cmd:option('-poolFrmsSize', 2, 'Number of emulator frames to pool over')
  cmd:option('-lifeLossTerminal', 'true', 'Use life loss as terminal signal (training only)')
  cmd:option('-flickering', 0, 'Probability of screen flickering (Catch only)')
  -- Experiment options
  cmd:option('-experiments', 'experiments', 'Base directory to store experiments')
  cmd:option('-_id', '', 'ID of experiment (used to store saved results, defaults to game name)')
  cmd:option('-network', '', 'Saved network weights file to load (weights.t7)')
  cmd:option('-verbose', 'false', 'Log info for every episode (only in train mode)')
  cmd:option('-saliency', '', 'Display saliency maps (requires QT): <none>|normal|guided|deconvnet')
  cmd:option('-record', 'false', 'Record screen (only in eval mode)')
  cmd:option('-evalRand', 'false', 'Whether to evaluate random policy')
  cmd:option('-trainWithRawData', 'false', 'Whether to use raw data in training, instead of using simulated player model')
  -- CI User Simulation Model Options
  cmd:option('-prepro', 'std', 'input state feature preprocessing: rsc | std')
  cmd:option('-ubgDir', 'ubgModel', 'directory storing uap and usp models')
  cmd:option('-uapFile', 'uap.t7', 'file storing userActsPredictor model')
  cmd:option('-uspFile', 'usp.t7', 'file storing userScorePredictor model')
  cmd:option('-actSmpLen', 8, 'The sampling candidate list length for user action generation')
  cmd:option('-actSmpEps', 0, 'User action sampling threshold. If rand se than this value, reture 1st pred')
  cmd:option('-termActSmgLen', 50, 'The length above which user termination action would be highly probably sampled. The observed avg length is about 40')
  cmd:option('-termActSmgEps', 0.9, 'The probability which user termination action would be sampled after certain length')
  cmd:option('-rwdSmpEps', 0, 'User rwd sampling threshold. If rand se than this value, reture 1st pred')
  cmd:option('-ciActRndSmp', 0, 'The probability under which player action is uniformly randomly sampled')
  cmd:option('-ciRwdRndSmp', 0, 'The probability under which player score(outcome) is uniformly randomly sampled')
  cmd:option('-ciGroup2rwd', -1, 'Reward signal design at terminal for 2nd group (below nlg median). It can be either 0 or -1')
  cmd:option('-ciRwdStMxTemp', -1, 'The temperature hyper-param used in Softmax distribution re-approximation. This is used in Reward sampling. If this re-approximation is not used, and random sampling is used, this param should be set to -1')
  cmd:option('-ciActStMxTemp', 1, 'The temperature hyper-param used in Softmax distribution re-approximation. This is used in Action sampling. If this re-approximation is not used, and random sampling is used, this param should be set to 1')
  cmd:option('-uppModel', 'rnn_rhn', 'type of player simulation model for action prediction. Only uap model type')
  cmd:option('-uppModelUsp', 'cnn_uSimCnn_moe', 'type of player simulation model for score(outcome) prediction. Only usp model type')
  cmd:option('-uppModelRNNDom', 0, 'Only for uap model, it is an indicator of whether the model is an RNN model and uses dropout masks from outside of the model. 0 for not using outside mask. Otherwise, this number represents the number of gates used in RNN model')
  cmd:option('-uppModelRNNDomUsp', 0, 'Only for usp model, it is an indicator of whether the model is an RNN model and uses dropout masks from outside of the model. 0 for not using outside mask. Otherwise, this number represents the number of gates used in RNN model')
  cmd:option('-lstmHist', 10, 'History length in input state representation used only in uap (user action predictor), not usp anymore')
  cmd:option('-lstmHistUsp', 2, 'History length in input state representation used only in usp only')
  cmd:option('-uSimLstmBackLen', 3, 'The maximum step applied in btpp in lstm')
  cmd:option('-rnnHdSizeL1', 21, 'lstm hidden layer size')
  cmd:option('-rnnHdSizeL2', 0, 'LSTM hidden layer size in 2nd lstm layer')
  cmd:option('-rnnHdLyCnt', 3, 'number of lstm hidden layer. Default is 2 bcz only when rnnHdSizeL2 is not 0 this opt will be examined')
  cmd:option('-rnnHdSizeL1Usp', 21, 'For usp model, lstm hidden layer size')
  cmd:option('-rnnHdSizeL2Usp', 0, 'For usp model, LSTM hidden layer size in 2nd lstm layer')
  cmd:option('-rnnHdLyCntUsp', 2, 'For usp model, number of lstm hidden layer. Default is 2 bcz only when rnnHdSizeL2 is not 0 this opt will be examined')
  cmd:option('-uSimGru', 0, 'Whether to substitue lstm with gru (0 for using lstm, 1 for GRU)')
  cmd:option('-actDistT', 100, 'The temperature hyper-param used in softmax of importance sampling')
  cmd:option('-ciunet', 'rlLoad', 'This opt is used in constructing player simulators. Do not change this value if using RL')
  cmd:option('-save', 'upplogs', 'subdirectory to save logs')
  cmd:option('-ciuTType', 'train', 'tell userSimulator which part of corpus to use')
  cmd:option('-uSimShLayer', 0, 'Whether the lower layers in Action and Score prediction NNs are shared. If this value is 1, use shared layers')
  cmd:option('-uSimScSoft', 0, 'The criterion weight of the score regression module in UserScoreSoftPrediction model. The value of this param should be in [0,1]. When it is 0, Soft prediction is off, and UserScorePrediction script is utilized. This opt is used to indicate whether to use UserScoreSoftPredictor (with value > 0) or UserScorePrdictor (with 0 value)')
  cmd:option('-testSetDivSeed', 2, 'The default seed value when separating a test set from the dataset')
  cmd:option('-validSetDivSeed', 3, 'The default seed value when separating a validation set out from the training set')
  cmd:option('-trainTwoFoldSim', 0, 'If this item is 1, we train player simulation model using 50% of data, meaning constructing player sim model for 2-fold cross validation in DRL evaluation')


  local opt = cmd:parse(arg)

  -- Process boolean options (Torch fails to accept false on the command line)
  opt.cudnn = opt.cudnn == 'true'
  opt.duel = opt.duel == 'true'
  opt.rlnnLinear = opt.rlnnLinear == 'true'
  opt.recurrent = opt.recurrent == 'true'
  opt.discretiseMem = opt.discretiseMem == 'true'
  opt.doubleQ = opt.doubleQ == 'true'
  opt.reportWeights = opt.reportWeights == 'true'
  opt.fullActions = opt.fullActions == 'true'
  opt.lifeLossTerminal = opt.lifeLossTerminal == 'true'
  opt.verbose = opt.verbose == 'true'
  opt.record = opt.record == 'true'
  opt.noValidation = opt.noValidation == 'true'
  opt.isevaprt = opt.isevaprt == 'true'
  opt.evalRand = opt.evalRand == 'true'
  opt.trainWithRawData = opt.trainWithRawData == 'true'
  opt.ac_greedy = opt.ac_greedy == 'true'
  opt.actor_critic = (opt.async == 'A3C' or opt.async == 'PPO')
  -- this list might be extended in future. Jan 1, 2018
  opt.ac_relative_plc = opt.ac_relative_plc == 'true'

  -- Process boolean/enum options
  if opt.colorSpace == '' then opt.colorSpace = false end
  if opt.memPriority == '' then opt.memPriority = false end
  if opt.async == '' then opt.async = false end
  if opt.saliency == '' then opt.saliency = false end
  if opt.async then opt.gpu = 0 end -- Asynchronous agents are CPU-only

  -- Because recurrent module in async models utilizes FastLSTM, we need set histLen to 1 to make experience work correctly
  if opt.async and opt.recurrent then opt.histLen = 1 print('@Async Recurrent: Reset histLen to 1 for async recurrent models') end

  -- Set ID as env (plus game name) if not set
  if opt._id == '' then
    local envName = paths.basename(opt.env)
    if opt.game == '' then
      opt._id = envName
    else
      opt._id = envName .. '.' .. opt.game
    end
  end

  -- set the uppModelRNNDom indicator in opt, which indicates whether the model is an RNN model, and uses dropout mask from outside the model construction
  -- right now, the rhn model, and Bayesian lstm model (following Gal's implementation), and GridLSTM model use outside dropout mask
  -- In this RL Setup opt construction, uppModelRNNDom is only for uap (user action predictor) model. Usp uses another opt item
  if string.sub(opt.uppModel, 1, 7) == 'rnn_rhn' then
    -- rnn_rhn uses double-sized dropout mask to drop out inputs of calculation of t-gate and transformed inner cell state
    opt.uppModelRNNDom = 2
  elseif string.sub(opt.uppModel, 1, 9) == 'rnn_blstm' or string.sub(opt.uppModel, 1, 13) == 'rnn_bGridlstm' then
    -- lstm used quad-sized dropout mask to drop out inputs of calculation of the 3 gates and transformed inner cell state
    opt.uppModelRNNDom = 4
  else
    opt.uppModelRNNDom = 0
  end

  -- Usp RNN dropout mask setup
  if string.sub(opt.uppModelUsp, 1, 7) == 'rnn_rhn' then
    -- rnn_rhn uses double-sized dropout mask to drop out inputs of calculation of t-gate and transformed inner cell state
    opt.uppModelRNNDomUsp = 2
  elseif string.sub(opt.uppModelUsp, 1, 9) == 'rnn_blstm' or string.sub(opt.uppModelUsp, 1, 13) == 'rnn_bGridlstm' then
    -- lstm used quad-sized dropout mask to drop out inputs of calculation of the 3 gates and transformed inner cell state
    opt.uppModelRNNDomUsp = 4
  else
    opt.uppModelRNNDomUsp = 0
  end

  -- Create one environment to extract specifications
  local Env = require(opt.env)
  local env = Env(opt)
  opt.stateSpec = env:getStateSpec()
  opt.actionSpec = env:getActionSpec()
  -- Process display if available (can be used for saliency recordings even without QT)
  if env.getDisplay then
    opt.displaySpec = env:getDisplaySpec()
  end

  return opt
end

-- Logs and aborts on error
local function abortIf(err, msg)
  if err then
    log.error(msg)
    error(msg)
  end
end

-- Validates setup options
function Setup:validateOptions()
  -- Check environment state is a single tensor
  abortIf(#self.opt.stateSpec ~= 3 or not _.isArray(self.opt.stateSpec[2]), 'Environment state is not a single tensor')

  -- Check environment has discrete actions
  abortIf(self.opt.actionSpec[1] ~= 'int' or self.opt.actionSpec[2] ~= 1, 'Environment does not have discrete actions')

  -- Change state spec if resizing
  if self.opt.height ~= 0 then
    self.opt.stateSpec[2][2] = self.opt.height
  end
  if self.opt.width ~= 0 then
    self.opt.stateSpec[2][3] = self.opt.width
  end

  -- Check colour conversions
  if self.opt.colorSpace then
    abortIf(not _.contains({'y', 'lab', 'yuv', 'hsl', 'hsv', 'nrgb'}, self.opt.colorSpace), 'Unsupported colour space for conversion')
    abortIf(self.opt.stateSpec[2][1] ~= 3, 'Original colour space must be RGB for conversion')
    -- Change state spec if converting from colour to greyscale
    if self.opt.colorSpace == 'y' then
      self.opt.stateSpec[2][1] = 1
    end
  end

  -- Check start of learning occurs after at least one minibatch of data has been collected
  abortIf(self.opt.learnStart <= self.opt.batchSize, 'learnStart must be greater than batchSize')

  -- Check enough validation transitions will be collected before first validation
  abortIf(self.opt.valFreq <= self.opt.valSize, 'valFreq must be greater than valSize')

  -- Check prioritised experience replay options
  abortIf(self.opt.memPriority and not _.contains({'rank', 'proportional'}, self.opt.memPriority), 'Type of prioritised experience replay unrecognised')
  abortIf(self.opt.memPriority == 'proportional', 'Proportional prioritised experience replay not implemented yet') -- TODO: Implement

  -- Check start of learning occurs after at least 1/100 of memory has been filled
  abortIf(self.opt.learnStart <= self.opt.memSize/100, 'learnStart must be greater than memSize/100')

  -- Check memory size is multiple of 100 (makes prioritised sampling partitioning simpler)
  abortIf(self.opt.memSize % 100 ~= 0, 'memSize must be a multiple of 100')

  -- Check learning occurs after first progress report
  abortIf(self.opt.learnStart < self.opt.progFreq, 'learnStart must be greater than progFreq')

  -- Check saliency map options
  abortIf(self.opt.saliency and not _.contains({'normal', 'guided', 'deconvnet'}, self.opt.saliency), 'Unrecognised method for visualising saliency maps')

  -- Check saliency is valid
  abortIf(self.opt.saliency and not self.opt.displaySpec, 'Saliency cannot be shown without env:getDisplay()')
  abortIf(self.opt.saliency and #self.opt.stateSpec[2] ~= 3 and (self.opt.stateSpec[2][1] ~= 3 or self.opt.stateSpec[2][1] ~= 1), 'Saliency cannot be shown without visual state')

  -- Check async options
  if self.opt.async then
    abortIf(self.opt.PALpha > 0, 'Persistent advantage learning not supported in async modes yet')
    abortIf(self.opt.bootstraps > 0, 'Bootstrap heads not supported in async mode yet')
    abortIf(self.opt.actor_critic and self.opt.duel, 'Dueling networks and actor-critic models are incompatible')
    abortIf(self.opt.actor_critic and self.opt.doubleQ, 'Double Q-learning and actor-critic models are incompatible')
    abortIf(self.opt.saliency, 'Saliency maps not supported in async modes yet')
  end

  -- Check CI player simulation modeling setting
  abortIf(self.opt.lstmHist < self.opt.lstmHistUsp, 'In CI user simulation modeling, It should be uap history length >= usp history length')
  abortIf(self.opt.ciActStMxTemp == 0, 'ciActStMxTemp should not be 0')
end

-- Augments environments with extra methods if missing
function Setup:augmentEnv()
  local Env = require(self.opt.env)
  local env = Env(self.opt)

  -- Set up fake training mode (if needed)
  if not env.training then
    Env.training = function() end
  end
  -- Set up fake evaluation mode (if needed)
  if not env.evaluate then
    Env.evaluate = function() end
  end
end

return Setup
