--
-- User: pwang8
-- Date: 1/31/17
-- Time: 11:48 PM
-- Just an identity layer added for CI, since our data is not image
--

local nn = require 'nn'
require 'classic.torch' -- Enables serialisation

local Body = classic.class('Body')

-- Constructor
function Body:_init(opts)
    opts = opts or {}

    self.recurrent = opts.recurrent -- The default value is false
    self.histLen = opts.histLen -- The default length value is 4
    self.stateSpec = opts.stateSpec
    self.rlnnLinear = false
    if next(opts) ~= nil then
        self.rlnnLinear = opts.rlnnLinear
    end
end

function Body:createBody()
    -- Number of input frames for recurrent networks is always 1
    local histLen = self.recurrent and 1 or self.histLen  -- If recurrent is true, then histLen is 1, else histLen is self.histLen
    local net = nn.Sequential()
    net:add(nn.View(histLen*self.stateSpec[2][1]*self.stateSpec[2][2]*self.stateSpec[2][3]))    -- nn.View() performs like Reshape()
    net:add(nn.Linear(histLen*self.stateSpec[2][1]*self.stateSpec[2][2]*self.stateSpec[2][3], 64))
    if not self.rlnnLinear then
        net:add(nn.ReLU(true))
    else
        print('Buidling up linear rl body!!!')
    end

    return net
end

return Body

