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

function RHN:__init(inputSize, outputSize, recurrence_depth, rhn_layers, rho)    -- p, mono in param list are deleted. not sure if mono is useful yet. mono is used in original lstm model to set Dropout
    parent.__init(self, rho or 9999)
    --self.p = p or 0   -- the param p and mono are not used right now, because we are implementing dropout outside of the RHN model,
    --self.mono = mono or false     -- it means dropout masks are passed into the RHN model as params
    self.inputSize = inputSize  -- It looks like output size does not have to equal to inputSize, because we did not
    self.outputSize = outputSize    -- do element-wise addition for x and hidden value directly. If we want to introduce residual module, then it is necessary
    self.recurrence_depth = recurrence_depth or 1    -- recurrence_depth in one RHN unit
    self.rhn_layers = rhn_layers or 1   -- this is the vertical layer number of the whole RHN. It is explicitly set up here because we want to use the output_rnn_dropout, which is not very convenient if used stacked structure in sequencer
    -- build the model
    self.recurrentModule = self:buildModel()
    -- make it work with nn.Container
    self.modules[1] = self.recurrentModule
    self.sharedClones[1] = self.recurrentModule

    -- for output(0), cell(0) and gradCell(T)
    self.zeroTensor = torch.Tensor()
    self.zeroCellTab = {}   -- we use a table to store hidden states from multiple layers in this multi-layer RHN model

    self.cells = {}
    self.gradCells = {}
end

-------------------------- factory methods -----------------------------

function RHN:buildRHNUnit(x, prev_h, noise_i, noise_h)
    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slice the n_gates dimension, i.e dimension 2
    local reshaped_noise_i = nn.Reshape(2, self.inputSize)(noise_i)   -- this might mean rhn has 2 gates, and this is the noise mask for input
    local reshaped_noise_h = nn.Reshape(2, self.outputSize)(noise_h)   -- this should be the noise for prior hidden state
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
                i2h[i]                  = nn.Linear(self.inputSize, self.outputSize)(dropped_x)    -- there are two i2h and h2h_tab bcz in equation 7 and 8 x and hidden state_h are utilized twice (2 sets of matrix multiplication)
                h2h_tab[layer_i][i]     = nn.Linear(self.outputSize, self.outputSize)(dropped_h_tab[layer_i])
            end
            t_gate_tab[layer_i]       = nn.Sigmoid()(nn.AddConstant(-2, False)(nn.CAddTable()({i2h[1], h2h_tab[layer_i][1]}))) -- this is the tranform module in equation 8 in the paper. I guess the AddConstant is an init step
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
                h2h_tab[layer_i][i]     = nn.Linear(self.outputSize, self.outputSize)(dropped_h_tab[layer_i]) -- h2h_tab[layer_i][1] is the multiplication in equation 8, h2h_tab[layer_i][2] is multiplication in equation 7
            end
            t_gate_tab[layer_i]       = nn.Sigmoid()(nn.AddConstant(-2, False)(h2h_tab[layer_i][1]))   -- Attention: refer to the Deep Transition RNN figure in readme file to check the structure here    -- Equation 8
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
-- of the multi-layer RHN model, so its size equals self.rhn_layers. All values contains in this table are outputs
-- of each RHN layer (of the last recurrent depth RHN cell in one layer). Values in hidden_s(t) are not dropout-ed
--
-- input of this Model is like: {x, prev_s}. x is the input of the whole RHN model, and prev_s is a table contains
-- prior calculated hidden state_s values. input also contains noise_i, noise_h, noise_o, which are dropout masks
-- used in RHN model. I thought about it and did not figure out if we can embed the dropout mask generation inside this
-- buildModel() function. To construct the variational RNN structure, in which dropout mask is held unchanged through
-- layers and time for one batch, it is not clear to me how to construct such a shared weights dropout module.
-- todo:pwang8. Nov 14, 2017. Maybe it can be achieved by adopting clone() or share() to construct customized dropout with shared weights
function RHN:buildModel()
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

-- In this multi-layer RHN model, this input param should only be the
-- data point feature values, not including dropout mask noises.
function RHN:getHiddenState(step, input) -- this input param is only used to set size of self.zeroTensor
    step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
    local prevCell
    if step == 0 then
        if input then
            if input:dim() == 2 then
                self.zeroTensor:resize(input:size(1), self.outputSize):zero()
            else
                self.zeroTensor:resize(self.outputSize):zero()
            end
        end
        self.zeroCellTab = {}
        for i=1, self.rhn_layers do table.insert(self.zeroCellTab, self.zeroTensor:clone()) end
        prevCell = self.userPrevCell or self.cells[step] or self.zeroCellTab
    else
        -- previous cell of this module
        prevCell = self.cells[step]
    end
    return prevCell
end

-- I suspect if this function is actually invoked in this implementation
function RHN:setHiddenState(step, hiddenState)
    step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
    assert(torch.type(hiddenState) == 'table')  -- in this multi-layer RHN model, hidden state should be a table containing hidden states from all (vertical) rhn layers
    assert(#hiddenState == self.rhn_layers)

    -- previous hidden states of this module
    self.cells[step] = hiddenState -- in this multi-layer RHN, the hiddenState should be a table of hidden state values from each (vertical) layer
end

------------------------- forward backward -----------------------------
function RHN:updateOutput(input)
    -- Attention: because we also have dropout masks as inputs into the RHN nn structure,
    -- so the input param here should be a table contains {x, noise_i, noise_h, noise_o}
    -- And this actually is the required input format for invoking the forward function
    -- when utilizing this multi-layer RHN module
    local _inputX, _inputNoise_i, _inputNoise_h, _inputNoise_o = unpack(input)
    local prevCell = self:getHiddenState(self.step-1, _inputX)

    local output, cell  -- in this multi-layer RHN, cell is a table of hidden state values from all (vertical) RHN layers
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

    -- when the RHN layer number is 1, the cell will be unpacked into a single tensor, so need to wrap it back to a table. This is weird
    if(torch.type(cell) ~= 'table') then cell = {cell} end
    self.outputs[self.step] = output    -- this is dropped_o
    self.cells[self.step] = cell    -- in this multi-layer RHN model, cell is a table of hidden state values from all (vertical) layers

    self.output = output
    self.cell = cell
    -- Get a deep copy of the cell value for its usage in evaluation mode
    if self.train == false then
        self.output = output:clone()    -- newly added, hope it works correctly
        self.cell = rnn.recursiveNew(cell)
        rnn.recursiveCopy(self.cell, cell)
    end

    self.step = self.step + 1
    self.gradPrevOutput = nil
    self.updateGradInputStep = nil
    self.accGradParametersStep = nil
    -- note that we don't return the cell, just the output
    -- although the output of our multi-layer RHN model is two-part {dropped_o, nn.Identity()(next_s)},
    -- after calling the forward() of this whole model, the output is only dropped_o
    return self.output
end

function RHN:getGradHiddenState(step)
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

function RHN:setGradHiddenState(step, gradHiddenState)
    local _step = self.updateGradInputStep or self.step
    step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
    assert(torch.type(gradHiddenState) == 'table')  -- in this multi-layer RHN model, hidden state should be a table containing hidden states from all (vertical) rhn layers
    assert(#gradHiddenState == self.rhn_layers)
    self.gradCells[step] = gradHiddenState
end

-- This input, as a input from invoking the multi-layer RHN model, should be in form of {x, noise_i, noise_h, noise_o}
function RHN:_updateGradInput(input, gradOutput)
    assert(self.step > 1, "expecting at least one updateOutput")
    local step = self.updateGradInputStep - 1
    assert(step >= 1)

    -- set the output/gradOutput states of current Module
    local recurrentModule = self:getStepModule(step)

    -- backward propagate through this step
    local gradCell = self:getGradHiddenState(step)
    assert(gradCell and torch.type(gradCell) == 'table', 'Gradient of cell values in multi-layer RHN model is wrong.')

    self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], gradOutput)

    -- Attention: because we also have dropout masks as inputs into the RHN nn structure,
    -- so the input param here should be a table contains {x, noise_i, noise_h, noise_o}
    -- and the _input into the multi-layer RHN model, according to the structure defined
    -- in buildModel(), should be {x, prev_s, noise_i, noise_h, noise_o}, so we add prev_s
    -- as the 2nd item in the input table
    assert(#input == 4, 'Input dim for the Multi-layer RHN should be 4.')
    local _inputX, _inputNoise_i, _inputNoise_h, _inputNoise_o = unpack(input)
    local _inTab = {_inputX, self:getHiddenState(step-1), _inputNoise_i, _inputNoise_h, _inputNoise_o}
    -- table.insert is not used bcz I'm afraid _accGradParameters() will use the same input table again

    if #gradCell == 1 then gradCell = unpack(gradCell) end  -- This is even more weird. When RHN layer # is 1, which means #gradCell == 1, I have to unpack gradCell bcz torch says it expects a tensor, not a table. But the table works for 2-layer RHN, which is actually following the structure defined above
    local gradInputTable = recurrentModule:updateGradInput(_inTab, {gradOutput, gradCell})   -- updateGradInput(input, gradOutput), from https://github.com/torch/nn/blob/master/doc/module.md#updategradinputinput-gradoutput
    self:setGradHiddenState(step-1, gradInputTable[2])  -- use gradInputTable[2] bcz input into the multi-layer RHN model is {x, prev_s, noise_i, noise_h, noise_o}

    return gradInputTable[1]    -- return [1] bcz the actual input into this structure contains hidden states, and we only want to output grad over actual input
end

function RHN:_accGradParameters(input, gradOutput, scale)
    local step = self.accGradParametersStep - 1
    assert(step >= 1)

    -- set the output/gradOutput states of current Module
    local recurrentModule = self:getStepModule(step)

    -- backward propagate through this step
    --local inputTable = self:getHiddenState(step-1)
    --table.insert(inputTable, 1, input)
    assert(#input == 4, 'Input dim for the Multi-layer RHN should be 4.')
    local _inputX, _inputNoise_i, _inputNoise_h, _inputNoise_o = unpack(input)
    local _inTab = {_inputX, self:getHiddenState(step-1), _inputNoise_i, _inputNoise_h, _inputNoise_o}
    local _gradCell = self:getGradHiddenState(step)
    gradOutput = self._gradOutputs[step] or gradOutput
    recurrentModule:accGradParameters(_inTab, {gradOutput, _gradCell}, scale)
end

function RHN:clearState()
    self.zeroTensor:set()
    -- self.zeroCellTab = {}
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
        -- self.zeroCellTab = {}
    end
    return parent.type(self, type, ...)
end

function RHN:local_Dropout(input, noise)
    return nn.CMulTable()({input, noise})
end

-- todo:pwang8. Nov 14, 2017. I guess I need a forget() to clear and reset dropout noise. Not sure if it is correct right now
