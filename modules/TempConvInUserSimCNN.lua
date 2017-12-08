------------------------------------------------------------------------
--[[ Temporal Convolution Module used in Player Simulation Modeling ]]--
-- Created by pwang8.
-- DateTime: 12/4/17 2:37 PM
-- Implemented following the OpenNMT lib from Stanford NLP Group
-- https://github.com/OpenNMT/OpenNMT
-- The NN module follows the construction of CNNEncoder
-- Expects 3D input.
-- For 3D input, the 1st dim is batch index, the 2nd dim is number of frames,
-- 3rd dim is number of features in one frame. For temporal CNN, one frame
-- represents information at one time step.
------------------------------------------------------------------------
assert(not nn.TempConvUserSimCNN, "update nnx package : luarocks install nnx")
local TempConvUserSimCNN, parent = torch.class('nn.TempConvUserSimCNN', 'nn.Module')
require 'modules.TemporalBatchNormalizationW'

------------------------------------------------------------------------
-------------------------------- Params --------------------------------
--- inputSize: input size in each frame into the CNN model
--- outputSize: output size in each frame from the CNN model
--- cnn_layers: the 'general' CNN layer counting. In v3, actually cnn layers will be doubled
--- kenal_width: kenrel width of each CNN module. All layers here use the same sized CNN module
--- frame_cnt: number of frames of input (and output, because we don't change it from input to output)
--- version: can be v1, v2, v3, or v4.
--- v1 has no residual connection,
--- v2 has residual connection for each CNN layer,
--- v3 has residual connects through 2 hidden CNN layers,
--- v4 has residual connection from the original input to each hidden CNN layer.
--- We only set kernel stride to be 1.
function TempConvUserSimCNN:__init()
    parent.__init(self)
end

function TempConvUserSimCNN:CreateCNNModule(inputSize, outputSize, cnn_layers, kernel_width, dropout_rate, frame_cnt, version)
    assert(version == 'v1' or version == 'v2' or version == 'v3' or version == 'v4', 'Convolution module in player simulation modeling can only be v1, v2 v3 or v4')
    assert(inputSize == outputSize, 'Right now we only support CNN modules with same input, output size for ease of adding residual connection')
    assert(dropout_rate >= 0 and dropout_rate < 1, 'Dropout rate should be in [0,1)')

    local input = nn.Identity()()
    local outputs = {}
    local _dropped_in = input
    if version == 'v4' and dropout_rate > 0 then _dropped_in = nn.Dropout(dropout_rate)(input) end

    for layer_idx=1, cnn_layers do
        local _input = input
        if layer_idx > 1 then _input = outputs[#outputs] end
        if dropout_rate > 0 then _input = nn.Dropout(dropout_rate)(_input) end
        local _pad = nn.Padding(1, kernel_width-1, 2)(_input) -- we set nInputDim of nn.Padding to be 2 dims(frame number, frame size), so input of 3-d will be treated as 1st dim being batch index
        local _conv = nn.TemporalConvolution(inputSize, outputSize, kernel_width)(_pad)
        local _batchNorm = nn.TemporalBatchNormalizationW(frame_cnt)(_conv)
        local _outConv = _batchNorm     -- v1
        if version == 'v2' then
            _outConv = nn.CAddTable()({_batchNorm, _input}) -- residual connection added for each convolution layer for version v2: added with value before convolution
        elseif version == 'v3' then
            local _nonlinear_1 = nn.ReLU()(_batchNorm)
            inputSize = outputSize  -- reset inputSize to make it usable in next layer construction
            if dropout_rate > 0 then _nonlinear_1 = nn.Dropout(dropout_rate)(_nonlinear_1) end
            local _pad_2 = nn.Padding(1, kernel_width-1, 2)(_nonlinear_1) -- we set nInputDim of nn.Padding to be 2 dims(frame number, frame size), so input of 3-d will be treated as 1st dim being batch index
            local _conv_2 = nn.TemporalConvolution(inputSize, outputSize, kernel_width)(_pad_2)
            local _batchNorm_2 = nn.TemporalBatchNormalizationW(frame_cnt)(_conv_2)
            _outConv = nn.CAddTable()({_batchNorm_2, _input}) -- residual connection added for each convolution layer for version v2
        elseif version == 'v4' then
            _outConv = nn.CAddTable()({_batchNorm, _dropped_in}) -- residual connection added for each convolution layer for version v4: direct adding with original input
        end
        local _output = nn.ReLU()(_outConv)
        table.insert(outputs, _output)
        inputSize = outputSize      -- reset inputSize to make it usable in next layer construction
    end

    return nn.gModule({input}, {outputs[#outputs]})
end