------------------------------------------------------------------------
--[[ Grid LSTM with Dropout in Gal's Variational RNN]]--
-- Created by pwang8.
-- DateTime: 11/27/17 5:32 PM
-- Implemented following the repo https://github.com/coreylynch/grid-lstm
-- and Yarin Gal's Bayesian LSTM implementation (for dropout implementation)
-- References:
-- Kalchbrenner, N., Danihelka, I., & Graves, A. (2015). Grid long short-term memory. arXiv preprint arXiv:1507.01526.
-- Gal, Y, "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks", 2015.
-- Zaremba, W, Sutskever, I, Vinyals, O, "Recurrent neural network regularization", 2014.
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for hidden state

-- It uses variational dropouts [Gal, 2015]. But dropout implementation is
-- based on dropout masks passed into this RNN model as params.
------------------------------------------------------------------------
assert(not nn.BayesianGridLSTM, "update nnx package : luarocks install nnx")
local BayesianGridLSTM, parent = torch.class('nn.BayesianGridLSTM', 'nn.AbstractRecurrent')

function BayesianGridLSTM:__init(rnn_size, rnn_layers, rho, tie_weights)    -- p, mono in param list are deleted. not sure if mono is useful yet. mono is used in original lstm model to set Dropout
    -- Attention: we assume this GridLSTM must have same dimension for input and output/hidden/cell values
    -- This is necessary because we need to add hidden unit values from depth dim and temporal dim, which
    -- requires depth hidden unit and temporal hidden unit have same dimension. Temporal hidden units have same size,
    -- so depth hidden units also have same size.
    parent.__init(self, rho or 9999)
    --self.p = p or 0   -- the param p and mono are not used right now, because we are implementing dropout outside of the RNN model,
    --self.mono = mono or false     -- it means dropout masks are passed into the RNN model as params
    self.rnnSize = rnn_size
    self.rnn_layers = rnn_layers or 1   -- this is the vertical layer number of the whole RNN/lstm. It is explicitly set up here because we want to use the output_rnn_dropout, which is not very convenient if used stacked structure in sequencer
    self.tie_weights = tie_weights or 1 -- whether NN weights along the depth dimension should be shared
    if self.tie_weights then
        self.GridLSTM_shared_weights = {}
        -- used in shared weights for gates and in_transform calculation along depth dim for hidden unit
        -- values from hidden along depth dim and hidden along temporal dim
        -- This implementation is different from that in Corey's repo, because we want to be able to
        -- apply different dropout masks on input(hidden) unit values for gate/in_transform calculation
        for i=1, 8 do table.insert(self.GridLSTM_shared_weights, nn.Linear(self.rnnSize, self.rnnSize)) end
    end
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

-------------------------- factory methods -----------------------------

--- Constructing the LSTM module in a GridLSTM. In the GridLSTM structure figure,
--- h_dep is the hidden value along depth dim (bottom), h_tem is the hidden value
--- along temporal dim (left). So, noise_i matches with h_dep, and noise_h maps
--- with h_tem.
function BayesianGridLSTM:buildBayesianLSTMUnit(h_dep, h_tem, prev_c, noise_i, noise_h, _shared_weights)
    -- In default, _shared_weights is off. In GridLSTM, _shared_weights is turned only along depth propagation.
    -- Temporal propagation does not need it because it naturally has shared weights in an RNN model.
    _shared_weights = _shared_weights or 0
    -- h_dep and noise_i should be a pair in dropout, h_tem and noise_h should be a pair in dropout
    -- In this GridLSTM implementation, the LSTM unit must have same dimension as input
    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slice the n_gates dimension, i.e dimension 2
    local reshaped_noise_i = nn.Reshape(4, self.rnnSize)(noise_i)   -- LSTM has 3 gates and 1 inner-cell value to calculate
    local reshaped_noise_h = nn.Reshape(4, self.rnnSize)(noise_h)
    local sliced_noise_i   = nn.SplitTable(2)(reshaped_noise_i) -- SplitTable(2) means split the input tensor along the 2nd dim, which is the num of gates dim
    local sliced_noise_h   = nn.SplitTable(2)(reshaped_noise_h)

    -- Calculate all 3 gates and in_transform
    local i2h, h2h         = {}, {} -- can be understood as calculation from depth(bottom) and temporal(left) hidden values
    for i = 1, 4 do
        -- Use select table to fetch each gate
        local dropped_x    = self:local_Dropout(h_dep, nn.SelectTable(i)(sliced_noise_i))
        local dropped_h    = self:local_Dropout(h_tem, nn.SelectTable(i)(sliced_noise_h))
        i2h[i]             = nn.Linear(self.rnnSize, self.rnnSize)(dropped_x)
        h2h[i]             = nn.Linear(self.rnnSize, self.rnnSize)(dropped_h)
    end

    -- See section 3.5, "Weight Sharing" of GridLSTM paper http://arxiv.org/pdf/1507.01526.pdf
    -- The weights along the temporal dimension are already tied (cloned in rnn training)
    -- Here we can tie the weights along the depth dimension. Having invariance in computation
    -- along the depth appears to be critical to solving the 15 digit addition problem w/ high accy.
    -- See fig 4. to compare tied vs untied grid lstms on this task.
    if _shared_weights == 1 then
        print("tying weights along the depth dimension in GridLSTM")
        for i=1, 4 do
            i2h[i].data.module:share(self.GridLSTM_shared_weights[i], 'weight', 'bias', 'gradWeight', 'gradBias')
            h2h[i].data.module:share(self.GridLSTM_shared_weights[i+4], 'weight', 'bias', 'gradWeight', 'gradBias')
        end
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

--- Attention: for Dropout implemented in this GridLSTM model, we utilize dropout masks from outside of the GridLSTM model
--- to implement the dropout in variational RNN model, as described in Gal's paper, which is different from Corey's implementation.
--- We only do dropout on hidden unit values, not on cell unit values.
--- For clearance, when saying hidden states/hidden cells along temporal dimension, we mean hidden and cell on left and right corner
--- in the 2d GridLSTM picture as in the paper. When saying hidden states/cells along depth dimension, we mean hidden and cell on
--- bottom and top in the the 2d GridLSTM picture.
function BayesianGridLSTM:buildModel()
    local x_dep            = nn.Identity()()    -- input to multi-layer GridLSTM, which is the hidden unit in 1st layer along depth dim (at bottom of the 2d GridLSTM)
    local cellZero_dep     = nn.Identity()()    -- cell unit value to multi-layer GridLSTM along depth dim, this is a placeholder which will always be a zero tensor
    local prev_s_tem       = nn.Identity()()    -- previous hidden state s and cell unit value c along temporal dimension (elements on left of a 2d GridLSTM) from each GridLSTM (depth) layer
    local noise_i          = nn.Identity()()    -- the dropout mask (before) entering the hidden layer. It quadruples the size of rnn_size, because it dropouts input with different patterns for calculation of 3 gates and in_transform. It applies on hidden values along depth dim. (Bottom of GridLSTM)
    local noise_h          = nn.Identity()()    -- the dropout mask for (before) the hidden layer. It quadruples the size of rnn_size for the same reason. This is used on hidden values along temporal dim. (Left of GridLSTM)
    local noise_o          = nn.Identity()()    -- dropout mask for the output of rnn/lstm (it's the output dropout mask of (after) the state_s on the highest RNN/lstm layer)
    --local i                = {[0] = x_dep}
    local next_s_tem       = {} -- the stored state_s (including hidden_tem, cell_tem) states along temporal dim for all depth (vertical) layers (on left/right in GridLSTM grid)
    local next_s_dep       = {} -- the stored state_s (including hidden_dep, cell_dep) states along depth dim (bottom/top in GridLSTM grid)
    local split            = {prev_s_tem:split(2 * self.rnn_layers)}  -- the split function is the split() for nngraph.Node. Can be found here: https://github.com/torch/nngraph/blob/master/node.lua (This is not the split function for tensor)
    local noise_i_split    = {noise_i:split(self.rnn_layers)} -- this nngraph.Node.split() function returns noutput number of new nodes that each take a single component of the output of this
    local noise_h_split    = {noise_h:split(self.rnn_layers)} -- node in the order they are returned.
    for layer_idx = 1, self.rnn_layers do     -- this self.rnn_layers is the vertical layer number of rnn/lstm, not the recurrent depth.
        local prev_c_tem         = split[2 * layer_idx - 1]     -- The prev_h is the hidden state_s value from previous time step along temporal dim (left on GridLSTM).
        local prev_h_tem         = split[2 * layer_idx]         -- Here it does not concern recurrent depth, which is sth studied inside rnn/lstm
        local n_i            = noise_i_split[layer_idx]     -- n_i and n_h are the dropout mask. n_i is the dropout mask for each (vertical) rnn/lstm layer's (vertical) input
        local n_h            = noise_h_split[layer_idx]     -- n_h is the dropout mask for each (horizontal) hidden unit in GridLSTM. Cell unit value does not apply dropout.
        local prev_c_dep
        local prev_h_dep
        if layer_idx == 1 then
            prev_c_dep = cellZero_dep  -- in 1st layer, the cell value along depth dim is a zero tensor. Should always be a zero tensor
            prev_h_dep = x_dep
        else
            prev_c_dep = next_s_dep[(layer_idx-1) * 2 - 1]
            prev_h_dep = next_s_dep[(layer_idx-1) * 2]
        end

        -- First calculate GridLSTM propagation along temporal dim
        local next_c_tem, next_h_tem = self:buildBayesianLSTMUnit(prev_h_dep, prev_h_tem, prev_c_tem, n_i, n_h, 0)
        table.insert(next_s_tem, next_c_tem)
        table.insert(next_s_tem, next_h_tem)
        --i[layer_idx] = next_h   -- this next_h is the state_s value, which is the output of one rnn/lstm module

        -- Calculate GridLSTM propagation along depth dim. Attention: next_h_tem is used in this step
        local next_c_dep, next_h_dep = self:buildBayesianLSTMUnit(prev_h_dep, next_h_tem, prev_c_dep, n_i, n_h, self.tie_weights)
        table.insert(next_s_dep, next_c_dep)
        table.insert(next_s_dep, next_h_dep)
    end
    local dropped_o = self:local_Dropout(next_s_dep[#next_s_dep], noise_o) -- the output of the whole multi-[vertical]layer GridLSTM, after dropout
    local module    = nn.gModule({x_dep, cellZero_dep, prev_s_tem, noise_i, noise_h, noise_o}, {dropped_o, nn.Identity()(next_s_tem)})
    module:getParameters():uniform(-0.04, 0.04)
    return module
end

-- In this multi-layer Baeysian GridLSTM model, this input should only be the
-- data point feature values, not including dropout mask noises.
-- For multi-layer Bayesian GridLSTM model, a hidden state should follow the format of
-- {prev_c_layer1, prev_h_layer1, prev_c_layer2, prev_h_layer2, ...}
function BayesianGridLSTM:getHiddenState(step, input) -- this input param is only used to set size of self.zeroTensor, by getting batch size
    step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
    --local prevOutput, prevCell
    local prevHidden
    if step == 0 then
        if input then
            if input:dim() == 2 then
                self.zeroTensor:resize(input:size(1), self.rnnSize):zero()
            else
                self.zeroTensor:resize(self.rnnSize):zero()
            end
        end
        self.zeroCellTab = {}
        -- Attention: we need 2 * rnn_layers of hidden units in table bcz we store prev_c and prev_h from each layer along temporal dim
        for i=1, 2*self.rnn_layers do table.insert(self.zeroCellTab, self.zeroTensor:clone()) end
        prevHidden = self.userPrevCell or self.cells[step] or self.zeroCellTab
    else
        -- previous cell of this module
        prevHidden = self.cells[step]
    end
    return prevHidden   -- the sequence is {L1_prev_c, L1_prev_h, L2_prev_c, L2_prev_h ...}
end

function BayesianGridLSTM:setHiddenState(step, hiddenState)
    step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
    assert(torch.type(hiddenState) == 'table')
    assert(#hiddenState == 2 * self.rnn_layers)   -- it is a table for prev_c and prev_h in each layer

    -- previous output of this module
    self.cells[step] = hiddenState
end

------------------------- forward backward -----------------------------
function BayesianGridLSTM:updateOutput(input)
    -- Attention: because we also have dropout masks as inputs into the RNN/lstm structure,
    -- so the input param here should be a table contains {x, noise_i, noise_h, noise_o}
    -- And this actually is the required input format for invoking the forward function
    -- when utilizing this multi-layer RNN module
    local _inputX, _inputNoise_i, _inputNoise_h, _inputNoise_o = unpack(input)
    local prevCell = self:getHiddenState(self.step-1, _inputX)
    local _zeroedCell = _inputX:clone():zero()  -- This is used in GridLSTM, which represents cell unit along depth dim for the 1st (bottom) layer. This should be a zero tensor

    -- {x_dep, cellZero_dep, prev_s_tem, noise_i, noise_h, noise_o}, {dropped_o, nn.Identity()(next_s_tem)}
    local output, cell
    if self.train ~= false then
        self:recycle()
        local recurrentModule = self:getStepModule(self.step)
        -- the actual forward propagation
        -- Attention: according to the model design in buildModel(), input should be {x_dep, prev_s_tem, noise_i, noise_h, noise_o}
        -- and prevOutput is not required in input param list.
        -- Output should be {dropped_o, (next_s_tem)}, here the next_s_tem stores all hidden/cell values along temporal dim.
        -- next_s_tem format is {layer1_cell_tem, layer1_hidden_tem, layer2_cell_tem, layer2_hidden_tem ...}
        output, cell = unpack(recurrentModule:updateOutput{_inputX, _zeroedCell, prevCell, _inputNoise_i, _inputNoise_h, _inputNoise_o})
    else
        if self.step==1 then prevCell = self.zeroCellTab else prevCell = self.cell end
        -- There was an error if prevCell is not set back to self.cell, only using value from self:getHiddenState
        -- in evaluation mode. And this problem is due to the design of nngraph and the recurrence mechanism in rnn lib.
        -- Simply, the invokation of updateOutput() in nngraph's gmodule will erase its input nodes, which might be the
        -- output of the same nn from prior time step. This problem is solved following post in this link:
        -- https://github.com/Element-Research/rnn/issues/172
        output, cell = unpack(self.recurrentModule:updateOutput{_inputX, _zeroedCell, prevCell, _inputNoise_i, _inputNoise_h, _inputNoise_o})
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

function BayesianGridLSTM:getGradHiddenState(step)
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

function BayesianGridLSTM:setGradHiddenState(step, gradHiddenState)
    local _step = self.updateGradInputStep or self.step
    step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
    assert(torch.type(gradHiddenState) == 'table')  -- in this multi-layer RNN/lstm model, hidden state should be a table containing hidden states from all (vertical) rnn/lstm layers
    assert(#gradHiddenState == 2 * self.rnn_layers)
    self.gradCells[step] = gradHiddenState
end

-- This input, as a input from invoking the multi-layer RNN/lstm model, should be in form of {x, noise_i, noise_h, noise_o}
function BayesianGridLSTM:_updateGradInput(input, gradOutput)
    assert(self.step > 1, "expecting at least one updateOutput")
    local step = self.updateGradInputStep - 1
    assert(step >= 1)

    -- set the output/gradOutput states of current Module
    local recurrentModule = self:getStepModule(step)

    -- backward propagate through this step
    local gradCell = self:getGradHiddenState(step)
    assert(gradCell and torch.type(gradCell) == 'table', 'Gradient of cell values in multi-layer Bayesian GridLSTM model is wrong.')

    self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], gradOutput)

    -- Attention: because we also have dropout masks as inputs into the RNN/lstm nn structure,
    -- so the input param here should be a table contains {x, noise_i, noise_h, noise_o}
    -- and the _input into the multi-layer RNN/lstm model, according to the structure defined
    -- in buildModel(), should be {x, prev_s, noise_i, noise_h, noise_o}, so we add prev_s
    -- as the 2nd item in the input table
    assert(#input == 4, 'Input dim for the Multi-layer Bayesian GridLSTM should be 4.')
    local _inputX, _inputNoise_i, _inputNoise_h, _inputNoise_o = unpack(input)
    local _zeroedCell = _inputX:clone():zero()  -- This is used in GridLSTM, which represents cell unit along depth dim for the 1st (bottom) layer. This should be a zero tensor
    local _inTab = {_inputX, _zeroedCell, self:getHiddenState(step-1), _inputNoise_i, _inputNoise_h, _inputNoise_o}
    -- table.insert is not used bcz I'm afraid _accGradParameters() will use the same input table again

    if #gradCell == 1 then gradCell = unpack(gradCell) end  -- This is even more weird. When RNN/lstm layer # is 1, which means #gradCell == 1, I have to unpack gradCell bcz torch says it expects a tensor, not a table. But the table works for 2-layer RNN/lstm, which is actually following the structure defined above
    local gradInputTable = recurrentModule:updateGradInput(_inTab, {gradOutput, gradCell})   -- updateGradInput(input, gradOutput), from https://github.com/torch/nn/blob/master/doc/module.md#updategradinputinput-gradoutput
    self:setGradHiddenState(step-1, gradInputTable[3])  -- use gradInputTable[3] for GridLSTM bcz input into the multi-layer RNN/lstm model is {x_dep, cellZero_dep, prev_s_tem, noise_i, noise_h, noise_o}, this is different from Bayesian LSTM implementation

    return gradInputTable[1]    -- return [1] bcz the actual input into this structure contains hidden states, and we only want to output grad over actual input
end

function BayesianGridLSTM:_accGradParameters(input, gradOutput, scale)
    local step = self.accGradParametersStep - 1
    assert(step >= 1)

    -- set the output/gradOutput states of current Module
    local recurrentModule = self:getStepModule(step)

    -- backward propagate through this step
    --local inputTable = self:getHiddenState(step-1)
    --table.insert(inputTable, 1, input)
    assert(#input == 4, 'Input dim for the Multi-layer Bayesian GridLSTM should be 4.')
    local _inputX, _inputNoise_i, _inputNoise_h, _inputNoise_o = unpack(input)
    local _zeroedCell = _inputX:clone():zero()  -- This is used in GridLSTM, which represents cell unit along depth dim for the 1st (bottom) layer. This should be a zero tensor
    local _inTab = {_inputX, _zeroedCell, self:getHiddenState(step-1), _inputNoise_i, _inputNoise_h, _inputNoise_o}
    local _gradCell = self:getGradHiddenState(step)
    gradOutput = self._gradOutputs[step] or gradOutput
    recurrentModule:accGradParameters(_inTab, {gradOutput, _gradCell}, scale)
end

function BayesianGridLSTM:clearState()
    self.zeroTensor:set()
    -- self.zeroCellTab = {}
    if self.userPrevOutput then self.userPrevOutput:set() end
    if self.userPrevCell then self.userPrevCell:set() end
    if self.userGradPrevOutput then self.userGradPrevOutput:set() end
    if self.userGradPrevCell then self.userGradPrevCell:set() end
    return parent.clearState(self)
end

function BayesianGridLSTM:type(type, ...)
    if type then
        self:forget()
        self:clearState()
        self.zeroTensor = self.zeroTensor:type(type)
        -- self.zeroCellTab = {}
    end
    return parent.type(self, type, ...)
end

function BayesianGridLSTM:local_Dropout(input, noise)
    return nn.CMulTable()({input, noise})
end