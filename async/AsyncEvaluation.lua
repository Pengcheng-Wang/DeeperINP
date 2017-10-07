local Display = require 'Display'
local ValidationAgent = require 'async/ValidationAgent'
local AsyncModel = require 'async/AsyncModel'
local classic = require 'classic'
local tds = require 'tds'

local AsyncEvaluation = classic.class('AsyncEvaluation')


function AsyncEvaluation:_init(opt)
  local asyncModel = AsyncModel(opt)
  local env = asyncModel:getEnvAndModel()
  local policyNet = asyncModel:createNet()
  local theta = policyNet:getParameters()

  if paths.filep(opt.network) then
    log.info('Loading pretrained network weights from ' .. opt.network)
    local weights = torch.load(opt.network)
    theta:copy(weights)
  else
    log.info('Loading pretrained network weights from last trained weights in ' .. opt._id)
    local weightsFile = paths.concat('experiments', opt._id, 'last.weights.t7')
    local weights = torch.load(weightsFile)
    theta:copy(weights)
  end

  local atomic = tds.AtomicCounter()
  self.validAgent = ValidationAgent(opt, theta, atomic)

  local state, adpType = env:start()
  self.hasDisplay = false
  if opt.displaySpec then
    self.hasDisplay = true
    self.display = Display(opt, env:getDisplay())
  end

  classic.strict(self)
end


function AsyncEvaluation:evaluate()
  local display = self.hasDisplay and self.display or nil
  self.validAgent:evaluate(display)
end

function AsyncEvaluation:ISevaluate()
  self.validAgent:ISevaluate()
end

return AsyncEvaluation
