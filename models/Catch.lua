local nn = require 'nn'
require 'classic.torch' -- Enables serialisation

local Body = classic.class('Body')

-- Constructor
function Body:_init(opts)
  opts = opts or {}

  self.recurrent = opts.recurrent -- The default value is false
  self.histLen = opts.histLen -- The default length value is 4
  self.stateSpec = opts.stateSpec
end

function Body:createBody()
  -- Number of input frames for recurrent networks is always 1
  local histLen = self.recurrent and 1 or self.histLen  -- If recurrent is true, then histLen is 1, else histLen is self.histLen
  local net = nn.Sequential()
  net:add(nn.View(histLen*self.stateSpec[2][1], self.stateSpec[2][2], self.stateSpec[2][3]))    -- nn.View() performs like Reshape()
  net:add(nn.SpatialConvolution(histLen*self.stateSpec[2][1], 32, 5, 5, 2, 2, 1, 1))
  net:add(nn.ReLU(true))
  net:add(nn.SpatialConvolution(32, 32, 5, 5, 2, 2))
  net:add(nn.ReLU(true))

  return net
end

return Body
