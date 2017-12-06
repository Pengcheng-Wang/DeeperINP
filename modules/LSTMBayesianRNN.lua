------------------------------------------------------------------------
--[[ Bayesian LSTM ]]--
-- Created by pwang8.
-- DateTime: 11/19/17 10:48 PM
-- Implemented following Yarin Gal's Bayesian LSTM implementation
-- References:
-- Gal, Y, "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks", 2015.
-- Zaremba, W, Sutskever, I, Vinyals, O, "Recurrent neural network regularization", 2014.
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for hidden state

-- It uses variational dropouts [Gal, 2015]. But dropout implementation is
-- based on dropout masks passed into this RNN model as params.
------------------------------------------------------------------------
assert(not nn.BayesianLSTM, "update nnx package : luarocks install nnx")
local BayesianLSTM, parent = torch.class('nn.BayesianLSTM', 'nn.AbstractRecurrent')

function BayesianLSTM:__init(inputSize, outputSize, rnn_layers, rho)    -- p, mono in param list are deleted. not sure if mono is useful yet. mono is used in original lstm model to set Dropout
    parent.__init(self, rho or 9999)
    --self.p = p or 0   -- the param p and mono are not used right now, because we are implementing dropout outside of the RNN model,
    --self.mono = mono or false     -- it means dropout masks are passed into the RNN model as params
    self.inputSize = inputSize  -- It looks like output size does not have to equal to inputSize, because we did not
    self.outputSize = outputSize    -- do element-wise addition for x and hidden value directly. If we want to introduce residual module, then it is necessary
    self.rnn_layers = rnn_layers or 1   -- this is the vertical layer number of the whole RNN/lstm. It is explicitly set up here because we want to use the output_rnn_dropout, which is not very convenient if used stacked structure in sequencer
    -- build the model
    self.recurrentModule = self:buildModel()
    -- make it work with nn.Container
    self.modules[1] = self.recurrentModule
    self.sharedClones[1] = self.recurrentModule

    -- for output(0), cell(0) and gradCell(T)
    self.zeroTensor = torch.Tensor()
    self.zeroCellTab = {}   -- we use a table to store hidden states from multiple layers in this multi-layer RNN model

    self.cells = {}
    self.gradCells = {}
end
-- todo:pwang8. Consider to add residual connection in BayesianLSTM, RHN and GridLSTM. Dec 6, 2017.
-------------------------- factory methods -----------------------------

function BayesianLSTM:buildBayesianLSTMUnit(x, prev_c, prev_h, noise_i, noise_h, _layerInd)
    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slice the n_gates dimension, i.e dimension 2
    local reshaped_noise_i
    local reshaped_noise_h = nn.Reshape(4, self.outputSize)(noise_h)
    if _layerInd == 1 then
        reshaped_noise_i = nn.Reshape(4, self.inputSize)(noise_i) -- LSTM has 3 gates and 1 inner-cell value to calculate
    else
        reshaped_noise_i = nn.Reshape(4, self.outputSize)(noise_i)
    end
    local sliced_noise_i   = nn.SplitTable(2)(reshaped_noise_i) -- SplitTable(2) means split the input tensor along the 2nd dim, which is the num of gates dim
    local sliced_noise_h   = nn.SplitTable(2)(reshaped_noise_h)
    -- Calculate all four gates
    local i2h, h2h         = {}, {}
    for i = 1, 4 do
        -- Use select table to fetch each gate
        local dropped_x      = self:local_Dropout(x, nn.SelectTable(i)(sliced_noise_i))
        local dropped_h      = self:local_Dropout(prev_h, nn.SelectTable(i)(sliced_noise_h))
        -- assumption: for more than 1 hidden layer LSTM models, we assume each layer has the same size
        if _layerInd == 1 then
            i2h[i]           = nn.Linear(self.inputSize, self.outputSize)(dropped_x)
        else
            i2h[i]           = nn.Linear(self.outputSize, self.outputSize)(dropped_x)
        end
        h2h[i]           = nn.Linear(self.outputSize, self.outputSize)(dropped_h)
    end

    -- Apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.CAddTable()({i2h[1], h2h[1]}))
    local in_transform     = nn.Tanh()(nn.CAddTable()({i2h[2], h2h[2]}))
    local forget_gate      = nn.Sigmoid()(nn.CAddTable()({i2h[3], h2h[3]}))
    local out_gate         = nn.Sigmoid()(nn.CAddTable()({i2h[4], h2h[4]}))

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return next_c, next_h
end

-- Attention: in lstm, we need to store 2 parts of hidden state, the cell value prev_c and hidden value prev_h
-- Following the implementation pattern from Gal's and Zaremba's code, I used a single node (prev_s) to store
-- both parts. In a multi-layer Bayesian lstm, prev_s is a table stores node values in format of
-- {prev_c_layer1, prev_h_layer1, prev_c_layer2, prev_h_layer2, ...}
-- The output of this multi-layer lstm module for the next_s has the same structure
-- Accordingly, getHiddenState() and setHiddenState() should following the same pattern for storing hidden states
function BayesianLSTM:buildModel()
    local x                = nn.Identity()()    -- input of rhn_network
    local prev_s           = nn.Identity()()    -- previous hidden state s from each lstm layer (including c and h outputs).
    local noise_i          = nn.Identity()()    -- the dropout mask (before) entering the hidden layer. It quadruples the size of rnn_size, because it dropouts input with different patterns for calculation of 3 gates and in_transform
    local noise_h          = nn.Identity()()    -- the dropout mask for (before) the hidden layer. It quadruples the size of rnn_size for the same reason
    local noise_o          = nn.Identity()()    -- dropout mask for the output of rnn/lstm (it's the output dropout mask of (after) the state_s on the highest RNN/lstm layer)
    local i                = {[0] = x}
    local next_s           = {} -- the stored state_s states for all RNN/lstm (vertical) layers
    local split            = {prev_s:split(2 * self.rnn_layers)}  -- the split function is the split() for nngraph.Node. Can be found here: https://github.com/torch/nngraph/blob/master/node.lua (This is not the split function for tensor)
    local noise_i_split    = {noise_i:split(self.rnn_layers)} -- this nngraph.Node.split() function returns noutput number of new nodes that each take a single component of the output of this
    local noise_h_split    = {noise_h:split(self.rnn_layers)} -- node in the order they are returned.
    for layer_idx = 1, self.rnn_layers do     -- this self.rnn_layers is the vertical layer number of rnn/lstm, not the recurrent depth.
        local prev_c         = split[2 * layer_idx - 1]
        local prev_h         = split[2 * layer_idx]         -- the prev_h is the hidden state_s value from previous time step. Here it does not concern recurrent depth, which is sth studied inside rnn/lstm
        local n_i            = noise_i_split[layer_idx]     -- n_i and n_h are the dropout mask. n_i is the dropout mask for each (vertical) rnn/lstm layer's (vertical) input
        local n_h            = noise_h_split[layer_idx]     -- n_h is the dropout mask for each (horizontal) rnn/lstm unit.
        local next_c, next_h = self:buildBayesianLSTMUnit(i[layer_idx - 1], prev_c, prev_h, n_i, n_h, layer_idx)
        table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        i[layer_idx] = next_h   -- this next_h is the state_s value, which is the output of one rnn/lstm module
    end
    local dropped_o          = self:local_Dropout(i[self.rnn_layers], noise_o)   -- the output of the whole (probably multi-[vertical]layer) RNN/lstm module, after dropout
    local module           = nn.gModule({x, prev_s, noise_i, noise_h, noise_o}, {dropped_o, nn.Identity()(next_s)})
    module:getParameters():uniform(-0.04, 0.04) -- this prev_s include {prev_c, prev_h} in each lstm layer
    return module
end

-- In this multi-layer Baeysian LSTM model, this input should only be the
-- data point feature values, not including dropout mask noises.
-- For multi-layer Bayesian lstm model, a hidden state should follow the format of
-- {prev_c_layer1, prev_h_layer1, prev_c_layer2, prev_h_layer2, ...}
function BayesianLSTM:getHiddenState(step, input) -- this input param is only used to set size of self.zeroTensor, by getting batch size
    step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
    --local prevOutput, prevCell
    local prevHidden
    if step == 0 then
        if input then
            if input:dim() == 2 then
                self.zeroTensor:resize(input:size(1), self.outputSize):zero()
            else
                self.zeroTensor:resize(self.outputSize):zero()
            end
        end
        self.zeroCellTab = {}
        -- Attention: we need 2 * rnn_layers of hidden units in table bcz we store prev_c and prev_h from each layer
        for i=1, 2*self.rnn_layers do table.insert(self.zeroCellTab, self.zeroTensor:clone()) end
        prevHidden = self.userPrevCell or self.cells[step] or self.zeroCellTab
    else
        -- previous cell of this module
        prevHidden = self.cells[step]
    end
    return prevHidden   -- the sequence is {L1_prev_c, L1_prev_h, L2_prev_c, L2_prev_h ...}
end

function BayesianLSTM:setHiddenState(step, hiddenState)
    step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
    assert(torch.type(hiddenState) == 'table')
    assert(#hiddenState == 2 * self.rnn_layers)   -- it is a table for prev_c and prev_h in each layer

    -- previous output of this module
    self.cells[step] = hiddenState
end

------------------------- forward backward -----------------------------
function BayesianLSTM:updateOutput(input)
    -- Attention: because we also have dropout masks as inputs into the RNN/lstm structure,
    -- so the input param here should be a table contains {x, noise_i, noise_h, noise_o}
    -- And this actually is the required input format for invoking the forward function
    -- when utilizing this multi-layer RNN module
    local _inputX, _inputNoise_i, _inputNoise_h, _inputNoise_o = unpack(input)
    local prevCell = self:getHiddenState(self.step-1, _inputX)

    -- {output(t), cell(t)} = lstm{x, prev_s, noise_i, noise_h, noise_o}
    local output, cell
    if self.train ~= false then
        self:recycle()
        local recurrentModule = self:getStepModule(self.step)
        -- the actual forward propagation
        -- Attention: according to the model design in buildModel(), input should be {x, prev_s, noise_i, noise_h, noise_o}
        -- and prevOutput is not required in input param list.
        -- and output should be {dropped_o, nn.Identity()(next_s)}
        output, cell = unpack(recurrentModule:updateOutput{_inputX, prevCell, _inputNoise_i, _inputNoise_h, _inputNoise_o})
    else
        if self.step==1 then prevCell = self.zeroCellTab else prevCell = self.cell end
        -- There was an error if prevCell is not set back to self.cell, only using value from self:getHiddenState
        -- in evaluation mode. And this problem is due to the design of nngraph and the recurrence mechanism in rnn lib.
        -- Simply, the invokation of updateOutput() in nngraph's gmodule will erase its input nodes, which might be the
        -- output of the same nn from prior time step. This problem is solved following post in this link:
        -- https://github.com/Element-Research/rnn/issues/172
        output, cell = unpack(self.recurrentModule:updateOutput{_inputX, prevCell, _inputNoise_i, _inputNoise_h, _inputNoise_o})
    end

    -- I've seen the case when cell only has one item and it will be unpacked into a single tensor. In lstm
    -- it might not be the problem bcz each hidden "cell" (word is not very accurate here) has two components,
    -- but still leave it here.
    if(torch.type(cell) ~= 'table') then cell = {cell} end
    self.outputs[self.step] = output    -- this is dropped_o
    self.cells[self.step] = cell

    self.output = output
    self.cell = cell
    -- Get a deep copy of the cell value for its usage in evaluation mode
    if self.train == false then
        self.output = output:clone()
        self.cell = rnn.recursiveNew(cell)
        rnn.recursiveCopy(self.cell, cell)
    end

    self.step = self.step + 1
    self.gradPrevOutput = nil
    self.updateGradInputStep = nil
    self.accGradParametersStep = nil
    -- note that we don't return the cell, just the output
    -- although the output of our multi-layer RNN/lstm model is two-part {dropped_o, nn.Identity()(next_s)},
    -- after calling the forward() of this whole model, the output is only dropped_o
    return self.output
end

function BayesianLSTM:getGradHiddenState(step)
    self.gradCells = self.gradCells or {}
    local _step = self.updateGradInputStep or self.step
    step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
    local gradCell
    if step == self.step-1 then
        gradCell = self.userNextGradCell or self.gradCells[step] or self.zeroCellTab
    else
        gradCell = self.gradCells[step]
    end
    return gradCell
end

function BayesianLSTM:setGradHiddenState(step, gradHiddenState)
    local _step = self.updateGradInputStep or self.step
    step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
    assert(torch.type(gradHiddenState) == 'table')  -- in this multi-layer RNN/lstm model, hidden state should be a table containing hidden states from all (vertical) rnn/lstm layers
    assert(#gradHiddenState == 2 * self.rnn_layers)
    self.gradCells[step] = gradHiddenState
end

-- This input, as a input from invoking the multi-layer RNN/lstm model, should be in form of {x, noise_i, noise_h, noise_o}
function BayesianLSTM:_updateGradInput(input, gradOutput)
    assert(self.step > 1, "expecting at least one updateOutput")
    local step = self.updateGradInputStep - 1
    assert(step >= 1)

    -- set the output/gradOutput states of current Module
    local recurrentModule = self:getStepModule(step)

    -- backward propagate through this step
    local gradCell = self:getGradHiddenState(step)
    assert(gradCell and torch.type(gradCell) == 'table', 'Gradient of cell values in multi-layer Bayesian LSTM model is wrong.')

    self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], gradOutput)

    -- Attention: because we also have dropout masks as inputs into the RNN/lstm nn structure,
    -- so the input param here should be a table contains {x, noise_i, noise_h, noise_o}
    -- and the _input into the multi-layer RNN/lstm model, according to the structure defined
    -- in buildModel(), should be {x, prev_s, noise_i, noise_h, noise_o}, so we add prev_s
    -- as the 2nd item in the input table
    assert(#input == 4, 'Input dim for the Multi-layer Bayesian LSTM should be 4.')
    local _inputX, _inputNoise_i, _inputNoise_h, _inputNoise_o = unpack(input)
    local _inTab = {_inputX, self:getHiddenState(step-1), _inputNoise_i, _inputNoise_h, _inputNoise_o}
    -- table.insert is not used bcz I'm afraid _accGradParameters() will use the same input table again

    if #gradCell == 1 then gradCell = unpack(gradCell) end  -- This is even more weird. When RNN/lstm layer # is 1, which means #gradCell == 1, I have to unpack gradCell bcz torch says it expects a tensor, not a table. But the table works for 2-layer RNN/lstm, which is actually following the structure defined above
    local gradInputTable = recurrentModule:updateGradInput(_inTab, {gradOutput, gradCell})   -- updateGradInput(input, gradOutput), from https://github.com/torch/nn/blob/master/doc/module.md#updategradinputinput-gradoutput
    self:setGradHiddenState(step-1, gradInputTable[2])  -- use gradInputTable[2] bcz input into the multi-layer RNN/lstm model is {x, prev_s, noise_i, noise_h, noise_o}

    return gradInputTable[1]    -- return [1] bcz the actual input into this structure contains hidden states, and we only want to output grad over actual input
end

function BayesianLSTM:_accGradParameters(input, gradOutput, scale)
    local step = self.accGradParametersStep - 1
    assert(step >= 1)

    -- set the output/gradOutput states of current Module
    local recurrentModule = self:getStepModule(step)

    -- backward propagate through this step
    --local inputTable = self:getHiddenState(step-1)
    --table.insert(inputTable, 1, input)
    assert(#input == 4, 'Input dim for the Multi-layer Bayesian LSTM should be 4.')
    local _inputX, _inputNoise_i, _inputNoise_h, _inputNoise_o = unpack(input)
    local _inTab = {_inputX, self:getHiddenState(step-1), _inputNoise_i, _inputNoise_h, _inputNoise_o}
    local _gradCell = self:getGradHiddenState(step)
    gradOutput = self._gradOutputs[step] or gradOutput
    recurrentModule:accGradParameters(_inTab, {gradOutput, _gradCell}, scale)
end

function BayesianLSTM:clearState()
    self.zeroTensor:set()
    -- self.zeroCellTab = {}
    if self.userPrevOutput then self.userPrevOutput:set() end
    if self.userPrevCell then self.userPrevCell:set() end
    if self.userGradPrevOutput then self.userGradPrevOutput:set() end
    if self.userGradPrevCell then self.userGradPrevCell:set() end
    return parent.clearState(self)
end

function BayesianLSTM:type(type, ...)
    if type then
        self:forget()
        self:clearState()
        self.zeroTensor = self.zeroTensor:type(type)
        -- self.zeroCellTab = {}
    end
    return parent.type(self, type, ...)
end

function BayesianLSTM:local_Dropout(input, noise)
    return nn.CMulTable()({input, noise})
end