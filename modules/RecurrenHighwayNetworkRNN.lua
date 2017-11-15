------------------------------------------------------------------------
--[[ Recurrent Highway Network ]]--
-- Created by pwang8.
-- DateTime: 11/14/17 10:19 AM
-- Recurrent Highway Network implemented following the Element-Research RNN lib format.
-- Ref. A.: https://arxiv.org/pdf/1607.03474.pdf
-- B. https://github.com/julian121266/RecurrentHighwayNetworks
-- C. https://github.com/Element-Research/rnn/blob/master/RHN.lua
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for hidden state

-- It uses variational dropouts [Gal, 2015]. But dropout implementation is
-- based on dropout masks passed into this RHN model as params.
------------------------------------------------------------------------
assert(not nn.RHN, "update nnx package : luarocks install nnx")
local RHN, parent = torch.class('nn.RHN', 'nn.AbstractRecurrent')

function RHN:__init(inputSize, recurrence_depth, rhn_layers, rho)    -- p, mono in param list are deleted. not sure if mono is useful yet. mono is used in original lstm model to set Dropout
    parent.__init(self, rho or 9999)
    --self.p = p or 0   -- the param p and mono are not used right now, because we are implementing dropout outside of the RHN model,
    --self.mono = mono or false     -- it means dropout masks are passed into the RHN model as params
    self.inputSize = inputSize  -- for RHN, the hiddensize should be the same as inputsize
    self.recurrence_depth = recurrence_depth    -- recurrence_depth in one RHN unit
    self.rhn_layers = rhn_layers or 1   -- this is the vertical layer number of the whole RHN. It is explicitly set up here because we want to use the output_rnn_dropout, which is not very convenient if used stacked structure in sequencer
    -- build the model
    self.recurrentModule = self:buildModel()
    -- make it work with nn.Container
    self.modules[1] = self.recurrentModule
    self.sharedClones[1] = self.recurrentModule

    -- for output(0), cell(0) and gradCell(T)
    self.zeroTensor = torch.Tensor()

    self.cells = {}
    self.gradCells = {}
end

-------------------------- factory methods -----------------------------

function RHN:buildRHNUnit(x, prev_h, noise_i, noise_h)
    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slice the n_gates dimension, i.e dimension 2
    local reshaped_noise_i = nn.Reshape(2, self.inputSize)(noise_i)   -- this might mean rhn has 2 gates, and this is the noise mask for input
    local reshaped_noise_h = nn.Reshape(2, self.inputSize)(noise_h)   -- this should be the noise for prior hidden state
    local sliced_noise_i   = nn.SplitTable(2)(reshaped_noise_i)   -- SplitTable(2) means split the input tensor along the 2nd dim, which is the num of gates dim
    local sliced_noise_h   = nn.SplitTable(2)(reshaped_noise_h)   -- after SplitTable, the output is a table of tensors
    -- Calculate all two gates
    local dropped_h_tab = {}
    local h2h_tab = {}
    local t_gate_tab = {}
    local c_gate_tab = {}
    local in_transform_tab = {}
    local s_tab = {}
    for layer_i = 1, self.recurrence_depth do -- Attention: pwang8. The for loop iteration count is the recurrence_depth (horizontal, in each time step), not RHN layers
        local i2h        = {}   -- i2h is a single item, bcz only one rhn unit is adopted to connect vertical layers (vertical input of x)
        h2h_tab[layer_i] = {}   -- h2h is of multiple items, bcz a large (10 in this example) recurrent depth is adopted (strictly it is not the (vertical) layers of NN, it is the horizontal depth defined in transition RNN in between one time step)
        if layer_i == 1 then
            for i = 1, 2 do
                -- Use select table to fetch each gate
                local dropped_x         = self:local_Dropout(x, nn.SelectTable(i)(sliced_noise_i)) -- slidced_noise_i is a table of tensors. So there are 2 gates and corresponding noise mask
                dropped_h_tab[layer_i]  = self:local_Dropout(prev_h, nn.SelectTable(i)(sliced_noise_h))  -- the 2 gates contain one gate for calc hidden state, and the other gate being the transform gate
                i2h[i]                  = nn.Linear(self.inputSize, self.inputSize)(dropped_x)    -- there are two i2h and h2h_tab bcz in equation 7 and 8 x and hidden state_h are utilized twice (2 sets of matrix multiplication)
                h2h_tab[layer_i][i]     = nn.Linear(self.inputSize, self.inputSize)(dropped_h_tab[layer_i])
            end
            t_gate_tab[layer_i]       = nn.Sigmoid()(nn.AddConstant(params.initial_bias, False)(nn.CAddTable()({i2h[1], h2h_tab[layer_i][1]}))) -- this is the tranform module in equation 8 in the paper. I guess the AddConstant is an init step
            in_transform_tab[layer_i] = nn.Tanh()(nn.CAddTable()({i2h[2], h2h_tab[layer_i][2]}))  -- calculate the hidden module, depicted in equation 7 in the paper
            c_gate_tab[layer_i]       = nn.AddConstant(1,false)(nn.MulConstant(-1, false)(t_gate_tab[layer_i])) -- in the implementation, the c gate is designed as (1-t), in which the t gate is calculated aboved
            s_tab[layer_i]           = nn.CAddTable()({
                nn.CMulTable()({c_gate_tab[layer_i], prev_h}),      -- Actually the input is not directly considered here. It's interesting to see how it performs if we add it
                nn.CMulTable()({t_gate_tab[layer_i], in_transform_tab[layer_i]})
            })  -- calc the output at time step t, as depicted in equation 6 in the paper
        else
            for i = 1, 2 do
                -- Use select table to fetch each gate
                dropped_h_tab[layer_i]  = self:local_Dropout(s_tab[layer_i-1], nn.SelectTable(i)(sliced_noise_h))
                h2h_tab[layer_i][i]     = nn.Linear(self.inputSize, self.inputSize)(dropped_h_tab[layer_i]) -- h2h_tab[layer_i][1] is the multiplication in equation 8, h2h_tab[layer_i][2] is multiplication in equation 7
            end
            t_gate_tab[layer_i]       = nn.Sigmoid()(nn.AddConstant(params.initial_bias, False)(h2h_tab[layer_i][1]))   -- Attention: refer to the Deep Transition RNN figure in readme file to check the structure here    -- Equation 8
            in_transform_tab[layer_i] = nn.Tanh()(h2h_tab[layer_i][2])  -- for transition layers inside one time step, only the h2h state values (horizontal) are propagated. So, it's a little different from the first transition layer   -- Equation 7
            c_gate_tab[layer_i]       = nn.AddConstant(1,false)(nn.MulConstant(-1, false)(t_gate_tab[layer_i]))     -- Equation 9, with the simplified assumption that c = 1 - t
            s_tab[layer_i]           = nn.CAddTable()({
                nn.CMulTable()({c_gate_tab[layer_i], s_tab[layer_i-1]}),
                nn.CMulTable()({t_gate_tab[layer_i], in_transform_tab[layer_i]})
            })  -- Equation 6
        end
    end
    local next_h = s_tab[self.recurrence_depth]
    return next_h   -- This is the output of one RHN unit, and this output has not been processed by Dropout
end

-- output of RHN:buildModel is table : {output(t), hidden_s(t)}. The whole model can be a multi-layer RHN,
-- so the output(t) is the output of the highest RHN (vertical) layer. And this output(t) is the result
-- after dropout using the mask of noise_o. hidden_s(t) is a table of hidden_s states for each (vertical) layer
-- of the multi-layer RHN model, so its size equals self.rhn_layers. All values contains in this table are values
-- of the each RHN layer (of the last recurrent depth RHN cell in one layer). Values in hidden_s(t) are not dropout-ed
--
-- input of this Model is like: {x, prev_s}. x is the input of the whole RHN model, and prev_s is a table contains
-- prior calculated hidden state_s values. input also contains noise_i, noise_h, noise_o, which are dropout masks
-- used in RHN model. I thought about it and did not figure out if we can embed the dropout mask generation inside this
-- buildModel() function. To construct the variational RNN structure, in which dropout mask is held unchanged through
-- layers and time for one batch, it is not clear to me how to construct such a shared weights dropout module.
-- todo:pwang8. Nov 14, 2017. Maybe it can be achieved by adopting clone() or share() to construct customized dropout with shared weights
function RHN:buildModel()
    -- todo:pwang8. The implementation is not correct! The current problem is that the input structrue of the nn is not set correctly. Input from outside of the model should contain x, and 3 noise mask. Pay attention how updateOutput() works.
    local x                = nn.Identity()()    -- input of rhn_network
    local prev_s           = nn.Identity()()    -- previous hidden state s from each rhn (vertical) layer.
    local noise_i          = nn.Identity()()    -- the dropout mask (before) entering the hidden layer. It doubles the size of rnn_size, bcz we use this input twice to calculate hidden state_s in rhn module and the t_gate.
    local noise_h          = nn.Identity()()    -- the dropout mask for (before) the hidden layer. It doubles the size of rnn_size, bcz we use this hidden state_h twice to calculate hidden state_s in rhn module and the t_gate.
    local noise_o          = nn.Identity()()    -- dropout mask for the output of rhn (it's the output dropout mask of (after) the state_s on the highest RHN layer)
    local i                = {[0] = x}
    local next_s           = {} -- the stored state_s states for all RHN (vertical) layers
    local split            = {prev_s:split(self.rhn_layers)}  -- the split function is the split() for nngraph.Node. Can be found here: https://github.com/torch/nngraph/blob/master/node.lua (This is not the split function for tensor)
    local noise_i_split    = {noise_i:split(self.rhn_layers)} -- this nngraph.Node.split() function returns noutput number of new nodes that each take a single component of the output of this
    local noise_h_split    = {noise_h:split(self.rhn_layers)} -- node in the order they are returned.
    for layer_idx = 1, self.rhn_layers do     -- this self.rhn_layers is the vertical layer number of rhn, not the recurrent depth.
        local prev_h         = split[layer_idx]         -- the prev_h is the hidden state_s value from previous time step. Here it does not concern recurrent depth, which is sth studied inside rhn
        local n_i            = noise_i_split[layer_idx]     -- n_i and n_h are the dropout mask. n_i is the dropout mask for each (vertical) rhn layer's (vertical) input
        local n_h            = noise_h_split[layer_idx]     -- n_h is the dropout mask for each (horizontal) rhn unit.
        local next_h = self:buildRHNUnit(i[layer_idx - 1], prev_h, n_i, n_h)
        table.insert(next_s, next_h)
        i[layer_idx] = next_h   -- this next_h is the state_s value, which is the output of one rhn module (may contain multiple recurrent depth)
    end
    local dropped_o          = self:local_Dropout(i[self.rhn_layers], noise_o)   -- the output of the whole (probably multi-[vertical]layer) RHN module, after dropout
    local module           = nn.gModule({x, prev_s, noise_i, noise_h, noise_o}, {dropped_o, nn.Identity()(next_s)})
    module:getParameters():uniform(-0.04, 0.04)
    return module
end

function RHN:getHiddenState(step, input) -- this input param is only used to set size of self.zeroTensor
    step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
    local prevOutput, prevCell
    if step == 0 then
        if input then
            if input:dim() == 2 then
                self.zeroTensor:resize(input:size(1), self.inputSize):zero()
            else
                self.zeroTensor:resize(self.inputSize):zero()
            end
        end
        prevOutput = self.userPrevOutput or self.outputs[step] or self.zeroTensor
        local _cellTab = {} -- Because this RHN model could have multiple layers, its hidden state_s is a table of all hidden states in all RHN layers at prior time step
        for i=1, self.rhn_layers do table.insert(_cellTab, self.zeroTensor:clone()) end
        prevCell = self.userPrevCell or self.cells[step] or _cellTab
    else
        -- previous output and cell of this module
        prevOutput = self.outputs[step]
        prevCell = self.cells[step]
    end
    return {prevOutput, prevCell}
end

function RHN:setHiddenState(step, hiddenState)
    step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
    assert(torch.type(hiddenState) == 'table')
    assert(#hiddenState == 2)

    -- previous output of this module
    self.outputs[step] = hiddenState[1]
    self.cells[step] = hiddenState[2]
end

------------------------- forward backward -----------------------------
function RHN:updateOutput(input)
    local prevOutput, prevCell = unpack(self:getHiddenState(self.step-1, input))

    -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
    local output, cell
    if self.train ~= false then
        self:recycle()
        local recurrentModule = self:getStepModule(self.step)
        -- the actual forward propagation
        output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
    else
        output, cell = unpack(self.recurrentModule:updateOutput{input, prevOutput, prevCell})
    end

    self.outputs[self.step] = output
    self.cells[self.step] = cell    -- cell is a table of tensors in RHN model, with each tensor inside the table represent hidden output value in each rhn_layer

    self.output = output
    self.cell = cell

    self.step = self.step + 1
    self.gradPrevOutput = nil
    self.updateGradInputStep = nil
    self.accGradParametersStep = nil
    -- note that we don't return the cell, just the output
    return self.output
end

function RHN:getGradHiddenState(step)
    self.gradOutputs = self.gradOutputs or {}
    self.gradCells = self.gradCells or {}
    local _step = self.updateGradInputStep or self.step
    step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
    local gradOutput, gradCell
    if step == self.step-1 then
        gradOutput = self.userNextGradOutput or self.gradOutputs[step] or self.zeroTensor
        gradCell = self.userNextGradCell or self.gradCells[step] or self.zeroTensor
    else
        gradOutput = self.gradOutputs[step]
        gradCell = self.gradCells[step]
    end
    return {gradOutput, gradCell}
end

function RHN:setGradHiddenState(step, gradHiddenState)
    local _step = self.updateGradInputStep or self.step
    step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
    assert(torch.type(gradHiddenState) == 'table')
    assert(#gradHiddenState == 2)

    self.gradOutputs[step] = gradHiddenState[1]
    self.gradCells[step] = gradHiddenState[2]
end

function RHN:_updateGradInput(input, gradOutput)
    assert(self.step > 1, "expecting at least one updateOutput")
    local step = self.updateGradInputStep - 1
    assert(step >= 1)

    -- set the output/gradOutput states of current Module
    local recurrentModule = self:getStepModule(step)

    -- backward propagate through this step
    local gradHiddenState = self:getGradHiddenState(step)
    local _gradOutput, gradCell = gradHiddenState[1], gradHiddenState[2]
    assert(_gradOutput and gradCell)

    self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], _gradOutput)
    nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
    gradOutput = self._gradOutputs[step]

    local inputTable = self:getHiddenState(step-1)
    table.insert(inputTable, 1, input)

    local gradInputTable = recurrentModule:updateGradInput(inputTable, {gradOutput, gradCell})   -- updateGradInput(input, gradOutput), from https://github.com/torch/nn/blob/master/doc/module.md#updategradinputinput-gradoutput

    local _ = require 'moses'
    self:setGradHiddenState(step-1, _.slice(gradInputTable, 2, 3)) -- use slice() bcz the inputTable contains 3 components, including {input(t), output(t), cell(t)}

    return gradInputTable[1]
end

function RHN:_accGradParameters(input, gradOutput, scale)
    local step = self.accGradParametersStep - 1
    assert(step >= 1)

    -- set the output/gradOutput states of current Module
    local recurrentModule = self:getStepModule(step)

    -- backward propagate through this step
    local inputTable = self:getHiddenState(step-1)
    table.insert(inputTable, 1, input)
    local gradOutputTable = self:getGradHiddenState(step)
    gradOutputTable[1] = self._gradOutputs[step] or gradOutputTable[1]
    recurrentModule:accGradParameters(inputTable, gradOutputTable, scale)
end

function RHN:clearState()
    self.zeroTensor:set()
    if self.userPrevOutput then self.userPrevOutput:set() end
    if self.userPrevCell then self.userPrevCell:set() end
    if self.userGradPrevOutput then self.userGradPrevOutput:set() end
    if self.userGradPrevCell then self.userGradPrevCell:set() end
    return parent.clearState(self)
end

function RHN:type(type, ...)
    if type then
        self:forget()
        self:clearState()
        self.zeroTensor = self.zeroTensor:type(type)
    end
    return parent.type(self, type, ...)
end

function RHN:local_Dropout(input, noise)
    return nn.CMulTable()({input, noise})
end

-- todo:pwang8. Nov 14, 2017. I guess I need a forget() to clear and reset dropout noise. Not sure if it is correct right now
