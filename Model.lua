local _ = require 'moses'
local paths = require 'paths'
local classic = require 'classic'
local nn = require 'nn'
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available
local nninit = require 'nninit'
local image = require 'image'
local DuelAggregator = require 'modules/DuelAggregator'
local TableSet = require 'MyMisc.TableSetMisc'
require 'classic.torch' -- Enables serialisation
require 'rnn'
require 'dpnn' -- Adds gradParamClip method
require 'modules/GuidedReLU'
require 'modules/DeconvnetReLU'
require 'modules/GradientRescale'
require 'modules/MinDim'

local Model = classic.class('Model')

-- Creates a Model (a helper for the network it creates)
function Model:_init(opt)
  -- Extract relevant options
  self.tensorType = opt.tensorType
  self.gpu = opt.gpu
  self.cudnn = opt.cudnn
  self.colorSpace = opt.colorSpace
  self.width = opt.width
  self.height = opt.height
  self.hiddenSize = opt.hiddenSize  -- Default value is 512. For the Catch demo, it is 32.
  self.histLen = opt.histLen
  self.duel = opt.duel  -- bool
  self.rlnnLinear = opt.rlnnLinear
  self.bootstraps = opt.bootstraps  -- int
  self.recurrent = opt.recurrent  -- bool
  self.env = opt.env  -- string. e.g., 'rlenvs.Catch'
  self.modelBody = opt.modelBody  -- string. e.g., 'models.Catch'
  self.async = opt.async  -- string. e.g., 'A3C' or 'NStepQ'
  self.stateSpec = opt.stateSpec
  self.opt = opt

  self.m = opt.actionSpec[3][2] - opt.actionSpec[3][1] + 1 -- Number of discrete actions
  -- Set up resizing
  if opt.width ~= 0 or opt.height ~= 0 then
    self.resize = true
    self.width = opt.width ~= 0 and opt.width or opt.stateSpec[2][3]
    self.height = opt.height ~= 0 and opt.height or opt.stateSpec[2][2]
  end
end

-- Processes a single frame for DQN input; must not return same memory to prevent side-effects
function Model:preprocess(observation)
  if self.opt.env == 'UserSimLearner/CIUserSimEnv' then
    -- In CI environment, we do not need to preprocess input feature set.
    return observation
  end

  local frame = observation:type(self.tensorType) -- Convert from CudaTensor if necessary

  -- Perform colour conversion if needed
  if self.colorSpace then
    frame = image['rgb2' .. self.colorSpace](frame)
  end

  -- Resize screen if needed
  if self.resize then
    frame = image.scale(frame, self.width, self.height)
  end

  -- Clone if needed
  if frame == observation then
    frame = frame:clone()
  end

  return frame
end

-- Calculates network output size
local function getOutputSize(net, inputDims)
  return net:forward(torch.Tensor(torch.LongStorage(inputDims))):size():totable()
end

-- Creates a DQN/AC model based on a number of discrete actions
function Model:create()
  -- Number of input frames for recurrent networks is always 1
  local histLen = self.recurrent and 1 or self.histLen

  -- Network starting with convolutional layers/model body
  local net = nn.Sequential()
  if self.recurrent then
    net:add(nn.Copy(nil, nil, true)) -- Needed when splitting batch x seq x input over seq for DRQN; better than nn.Contiguous
  end

  -- Add network body
  log.info('Setting up ' .. self.modelBody)
  local Body = require(self.modelBody)
  local body = Body(self):createBody()

  -- Calculate body output size
  local bodyOutputSize = torch.prod(torch.Tensor(getOutputSize(body, _.append({histLen}, self.stateSpec[2]))))  -- return of _.append({histLen}, self.stateSpec[2]) is a table of {4, 1, 24, 24} for Catch demo
  -- When _.append({histLen}, self.stateSpec[2]) is like {1, 1, 24, 24} in RNN setting on Catch, getOutputSize(...) is like {32, 4, 4} after the CNN processing (there should be another batchIndex)
  if not self.async and self.recurrent then --
    body:add(nn.View(-1, bodyOutputSize))
    net:add(nn.MinDim(1, 4))  -- If input dimension < 4, then add one extra dimension at index 1
    net:add(nn.Transpose({1, 2})) -- swap 1st and 2nd dimension -- This is used for SeqLSTM when recurrent is true and async is false. SeqLSTM requires batchIndex be the 2nd dim
    body = nn.Bottle(body, 4, 2)  -- Bottle allows varying dimensionality input to be forwarded through any module that accepts input of nInputDim dimensions, and generates output of nOutputDim dimensions.
    net:add(body)
    net:add(nn.MinDim(1, 3))
  else
     body:add(nn.View(bodyOutputSize))
     net:add(body)
  end
  -- print('###', getOutputSize(net, {50, 10, 1, 24, 24})) os.exit() -- The printed info is {10, 50, 512}. I suppose 50 is batchSize, 10 is histLen. 512 is bcz CNN's output is 32*4*4.
  -- The above is the situation when recurrent is true and async is false
  -- When recurrent is false, this printed info becomes {50, 512}

  -- Network head
  local head = nn.Sequential()
  local heads = math.max(self.bootstraps, 1)  -- for the Catch demo, the default bootstraps value is 0
  if self.rlnnLinear then
    head:add(nn.Linear(bodyOutputSize, self.m))
  elseif self.duel then -- In default, duel network is used
    -- Value approximator V^(s)
    local valStream = nn.Sequential()
    if self.recurrent and self.async then
      local lstm = nn.FastLSTM(bodyOutputSize, self.hiddenSize, self.opt.asyncRecrRho) -- the 3rd param, [rho], the maximum amount of backpropagation steps to take back in time, default value is 9999
      TableSet.fastLSTMForgetGateInit(lstm, 0, self.hiddenSize, nninit)
      lstm:remember('both')
      valStream:add(lstm)
    elseif self.recurrent then
      local lstm = nn.SeqLSTM(bodyOutputSize, self.hiddenSize)
      lstm:remember('both')
      valStream:add(lstm)
      valStream:add(nn.Select(-3, -1)) -- Select last timestep -- This is the reason why the output is of the same format even if recurrent is utilized. It's just because only output at last time step is used, by pwang8
    else
      valStream:add(nn.Linear(bodyOutputSize, self.hiddenSize))
      valStream:add(nn.ReLU(true))
    end
    valStream:add(nn.Linear(self.hiddenSize, 1)) -- Predicts value for state

    -- Advantage approximator A^(s, a)
    local advStream = nn.Sequential()
    if self.recurrent and self.async then
      local lstm = nn.FastLSTM(bodyOutputSize, self.hiddenSize, self.opt.asyncRecrRho)
      TableSet.fastLSTMForgetGateInit(lstm, 0, self.hiddenSize, nninit)
      lstm:remember('both')
      advStream:add(lstm)
    elseif self.recurrent then
      local lstm = nn.SeqLSTM(bodyOutputSize, self.hiddenSize)
      lstm:remember('both')
      advStream:add(lstm)
      advStream:add(nn.Select(-3, -1)) -- Select last timestep
    else
      advStream:add(nn.Linear(bodyOutputSize, self.hiddenSize))
      advStream:add(nn.ReLU(true))
    end
    advStream:add(nn.Linear(self.hiddenSize, self.m)) -- Predicts action-conditional advantage

    -- Streams container
    local streams = nn.ConcatTable()
    streams:add(valStream)
    streams:add(advStream)

    -- Network finishing with fully connected layers
    head:add(nn.GradientRescale(1/math.sqrt(2), true)) -- Heuristic that mildly increases stability for duel
    -- Create dueling streams
    head:add(streams)
    -- Add dueling streams aggregator module
    head:add(DuelAggregator(self.m))
    -- print('###', getOutputSize(head, {10, 50, 512})) os.exit() -- In case self.recurrent is true, input to one head is of size {10, 50, 512},
    -- with 10 being histLen, 50 being batchSize, and 512 be # of features from output of previous layers. Output is of size {50, 3, 1},
    -- with 3 being # of actions. If self.recurrent is false, input to one head module
    -- should be {50, 512}. Then output of head module is still {50, 3, 1}
  else
    if self.recurrent and self.async then
      local lstm = nn.FastLSTM(bodyOutputSize, self.hiddenSize, self.opt.asyncRecrRho)
      TableSet.fastLSTMForgetGateInit(lstm, 0, self.hiddenSize, nninit)
      lstm:remember('both')
      head:add(lstm)
    elseif self.recurrent then
      local lstm = nn.SeqLSTM(bodyOutputSize, self.hiddenSize)
      lstm:remember('both')
      head:add(lstm)  -- output here is of size {10, 50, 32}, with 10 being histLen, 50 being batchSize, 32 being hidden neuron #
      head:add(nn.Select(-3, -1)) -- Select last timestep. Output here is of size {50, 32}
      -- print('###', getOutputSize(head, {10, 50, 512})) os.exit()
    else
      head:add(nn.Linear(bodyOutputSize, self.hiddenSize))
      head:add(nn.ReLU(true)) -- DRQN paper reports worse performance with ReLU after LSTM
    end
    head:add(nn.Linear(self.hiddenSize, self.m)) -- Note: Tuned DDQN uses shared bias at last layer
  end

  if self.bootstraps > 0 then -- In Setup.lua the default value is 10. For Catch demo, the default value is 0
    -- Add bootstrap heads
    local headConcat = nn.ConcatTable()
    for h = 1, heads do
      -- Clone head structure
      local bootHead = head:clone()
      -- Each head should use a different random initialisation to construct bootstrap (currently Torch default)
      local linearLayers = bootHead:findModules('nn.Linear')
      for l = 1, #linearLayers do
        linearLayers[l]:init('weight', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)}):init('bias', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)})
      end
      headConcat:add(bootHead)
    end
    net:add(nn.GradientRescale(1/self.bootstraps)) -- Normalise gradients by number of heads
    net:add(headConcat)
  elseif self.opt.actor_critic then
    -- Actor-critic does not use the normal head but instead a concatenated value function V and policy π
    -- Actor-critic method is so differently implemented from other value function-based methods. The head module,
    -- right now including potentially bootstrapped heads, recurrent module, or dule module are not included in
    -- Actor-critic(e.g., A3C, PPO) methods.
    -- Because the theoretically meaning of dule module, dobule-Q and PAL is not compatible with actor_critic,
    -- we will not include them in actor_critic model construction. Recurrent module should be able to added though.
    -- Bootstrap is not considered right now for actor-critic models.
    if self.recurrent then
      local lstm = nn.FastLSTM(bodyOutputSize, self.hiddenSize, self.opt.asyncRecrRho)
      TableSet.fastLSTMForgetGateInit(lstm, 0, self.hiddenSize, nninit)   --Not sure if lazy dropout is correct. So not using dropout for recurrent module in DRL right now
      lstm:remember('both')
      net:add(lstm)
    else
      net:add(nn.Linear(bodyOutputSize, self.hiddenSize))
      if not self.opt.rlnnLinear then
        net:add(nn.ReLU(true))
      end
    end

    local valueAndPolicy = nn.ConcatTable() -- π and V share all layers except the last

    -- Value function V(s; θv)
    local valueFunction = nn.Linear(self.hiddenSize, 1)

    -- Policy π(a | s; θπ)
    local policy = nn.Sequential()
    policy:add(nn.Linear(self.hiddenSize, self.m))
    policy:add(nn.SoftMax())

    valueAndPolicy:add(valueFunction)
    valueAndPolicy:add(policy)

    net:add(valueAndPolicy)

    -- Each head should use a different random initialisation to construct bootstrap (currently Torch default)
    local _linearLayers = net:findModules('nn.Linear')
    for l = 1, #_linearLayers do
      _linearLayers[l]:init('weight', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)}):init('bias', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)})
    end
  else
    -- Add head via ConcatTable (simplifies bootstrap code in agent)
    local headConcat = nn.ConcatTable()
    headConcat:add(head)
    net:add(headConcat)
  end

  if not self.opt.actor_critic then
    net:add(nn.JoinTable(1, 1))
    net:add(nn.View(heads, self.m))
  end
  -- GPU conversion
  if self.gpu > 0 then
    require 'cunn'
    net:cuda()

    if self.cudnn and hasCudnn then
      cudnn.convert(net, cudnn)
      -- The following is legacy code that can make cuDNN deterministic (with a large drop in performance)
      --[[
      local convs = net:findModules('cudnn.SpatialConvolution')
      for i, v in ipairs(convs) do
        v:setMode('CUDNN_CONVOLUTION_FWD_ALGO_GEMM', 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1', 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')
      end
      --]]
    end
  end
  
  -- Save reference to network
  self.net = net

  return net
end

function Model:setNetwork(net)
  self.net = net
end

-- Return list of convolutional filters as list of images
function Model:getFilters()
  local filters = {}

  -- Find convolutional layers
  local convs = self.net:findModules(self.cudnn and hasCudnn and 'cudnn.SpatialConvolution' or 'nn.SpatialConvolution')
  for i, v in ipairs(convs) do
    -- Add filter to list (with each layer on a separate row)
    filters[#filters + 1] = image.toDisplayTensor(v.weight:view(v.nOutputPlane*v.nInputPlane, v.kH, v.kW), 1, v.nInputPlane, true)
  end

  return filters
end

-- Set ReLUs up for specified saliency visualisation type
function Model:setSaliency(saliency)
  -- Set saliency
  self.saliency = saliency

  -- Find ReLUs on existing model
  local relus, relucontainers = self.net:findModules(hasCudnn and 'cudnn.ReLU' or 'nn.ReLU')
  if #relus == 0 then
    relus, relucontainers = self.net:findModules('nn.GuidedReLU')
  end
  if #relus == 0 then
    relus, relucontainers = self.net:findModules('nn.DeconvnetReLU')
  end

  -- Work out which ReLU to use now
  local layerConstructor = hasCudnn and cudnn.ReLU or nn.ReLU
  self.relus = {} --- Clear special ReLU list to iterate over for salient backpropagation
  if saliency == 'guided' then
    layerConstructor = nn.GuidedReLU
  elseif saliency == 'deconvnet' then
    layerConstructor = nn.DeconvnetReLU
  end

  -- Replace ReLUs
  for i = 1, #relus do
    -- Create new special ReLU
    local layer = layerConstructor()

    -- Copy everything over
    for key, val in pairs(relus[i]) do
      layer[key] = val
    end

    -- Find ReLU in containing module and replace
    for j = 1, #(relucontainers[i].modules) do
      if relucontainers[i].modules[j] == relus[i] then
        relucontainers[i].modules[j] = layer
      end
    end
  end

  -- Create special ReLU list to iterate over for salient backpropagation
  self.relus = self.net:findModules(saliency == 'guided' and 'nn.GuidedReLU' or 'nn.DeconvnetReLU')
end

-- Switches the backward computation of special ReLUs for salient backpropagation
function Model:salientBackprop()
  for i, v in ipairs(self.relus) do
    v:salientBackprop()
  end
end

-- Switches the backward computation of special ReLUs for normal backpropagation
function Model:normalBackprop()
  for i, v in ipairs(self.relus) do
    v:normalBackprop()
  end
end

return Model
