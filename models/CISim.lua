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
    opts = opts.opt  --opts or {}

    self.recurrent = opts.recurrent -- The default value is false
    self.histLen = opts.histLen -- The default length value is 4
    self.stateSpec = opts.stateSpec
    self.rlnnLinear = false
    if next(opts) ~= nil then
        self.rlnnLinear = opts.rlnnLinear
    end
    self.ciTemCnn = opts.ciTemCnn   -- number of temporal cnn layers added in CI-DRL model
    if self.ciTemCnn > 0.5 then require 'modules.TempConvInUserSimCNN' end
    assert(not (self.ciTemCnn > 0.5 and self.recurrent), 'Right now we do not support recurrent cnn model in DRL for CI problem')
    self.opt = opts
end

function Body:createBody()
    -- Number of input frames for recurrent networks is always 1
    local histLen = self.recurrent and 1 or self.histLen  -- If recurrent is true, then histLen is 1, else histLen is self.histLen
    local net = nn.Sequential()
    if self.ciTemCnn > 0.5 then
        -- DRL network with temporal cnn
        assert(histLen > 1, 'histLen must > 1 if temporal cnn is required in DRL model')
        net:add(nn.Squeeze())   -- Input into this module should be 4-d tensor representing {histLen, stateSpec[2][1], stateSpec[2][2], stateSpec[2][3]}. For CI, stateSpec[2][1] == 1 and stateSpec[2][2] == 1
        local tempCnn = nn.TempConvUserSimCNN()         -- inputSize, outputSize, cnn_layers, kernel_width, dropout_rate, version
        local _tempCnnLayer = tempCnn:CreateCNNModule(self.stateSpec[2][3], self.stateSpec[2][3], self.opt.ciTemCnn, self.opt.drlCnnKernelWidth, self.opt.rlDropout, histLen, self.opt.drlCnnConnType)
        net:add(_tempCnnLayer)
    else
        -- DRL network without temporal cnn
        net:add(nn.View(histLen*self.stateSpec[2][1]*self.stateSpec[2][2]*self.stateSpec[2][3]))    -- nn.View() performs like Reshape()
        net:add(nn.Linear(histLen*self.stateSpec[2][1]*self.stateSpec[2][2]*self.stateSpec[2][3], 64))
        if not self.rlnnLinear then
            net:add(nn.ReLU(true))
        else
            print('Buidling up linear rl body!!!')
        end
    end

    return net
end

return Body

