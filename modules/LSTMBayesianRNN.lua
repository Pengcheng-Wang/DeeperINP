------------------------------------------------------------------------
--[[ Bayesian LSTM ]]--
-- Created by pwang8.
-- DateTime: 11/19/17 10:48 PM
-- implemented following Yarin Gal's Bayesian LSTM implementation
-- References:
-- Gal, Y, "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks", 2015.
-- Zaremba, W, Sutskever, I, Vinyals, O, "Recurrent neural network regularization", 2014.
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for hidden state

-- It uses variational dropouts [Gal, 2015]. But dropout implementation is
-- based on dropout masks passed into this RHN model as params.
------------------------------------------------------------------------
assert(not nn.BayesianLSTM, "update nnx package : luarocks install nnx")
local BayesianLSTM, parent = torch.class('nn.BayesianLSTM', 'nn.AbstractRecurrent')

function BayesianLSTM:__init(inputSize, outputSize, rhn_layers, rho)    -- p, mono in param list are deleted. not sure if mono is useful yet. mono is used in original lstm model to set Dropout
    parent.__init(self, rho or 9999)
    --self.p = p or 0   -- the param p and mono are not used right now, because we are implementing dropout outside of the RHN model,
    --self.mono = mono or false     -- it means dropout masks are passed into the RHN model as params
    self.inputSize = inputSize  -- It looks like output size does not have to equal to inputSize, because we did not
    self.outputSize = outputSize    -- do element-wise addition for x and hidden value directly. If we want to introduce residual module, then it is necessary
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

function BayesianLSTM:buildBayesianLSTMUnit(x, prev_c, prev_h, noise_i, noise_h)
    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slice the n_gates dimension, i.e dimension 2
    local reshaped_noise_i = nn.Reshape(4, self.inputSize)(noise_i)
    local reshaped_noise_h = nn.Reshape(4, self.inputSize)(noise_h)
    local sliced_noise_i   = nn.SplitTable(2)(reshaped_noise_i)
    local sliced_noise_h   = nn.SplitTable(2)(reshaped_noise_h)
    -- Calculate all four gates
    local i2h, h2h         = {}, {}
    for i = 1, 4 do
        -- Use select table to fetch each gate
        local dropped_x      = local_Dropout(x, nn.SelectTable(i)(sliced_noise_i))
        local dropped_h      = local_Dropout(prev_h, nn.SelectTable(i)(sliced_noise_h))
        i2h[i]               = nn.Linear(self.inputSize, self.outputSize)(dropped_x)
        h2h[i]               = nn.Linear(self.inputSize, self.outputSize)(dropped_h)
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

function BayesianLSTM:buildModel()
    --local x                = nn.Identity()()    -- input of rhn_network
    --local prev_s           = nn.Identity()()    -- previous hidden state s from each rhn (vertical) layer.
    --local noise_i          = nn.Identity()()    -- the dropout mask (before) entering the hidden layer. It doubles the size of rnn_size, bcz we use this input twice to calculate hidden state_s in rhn module and the t_gate.
    --local noise_h          = nn.Identity()()    -- the dropout mask for (before) the hidden layer. It doubles the size of rnn_size, bcz we use this hidden state_h twice to calculate hidden state_s in rhn module and the t_gate.
    --local noise_o          = nn.Identity()()    -- dropout mask for the output of rhn (it's the output dropout mask of (after) the state_s on the highest RHN layer)
    --local i                = {[0] = x}
    --local next_s           = {} -- the stored state_s states for all RHN (vertical) layers
    --local split            = {prev_s:split(self.rhn_layers)}  -- the split function is the split() for nngraph.Node. Can be found here: https://github.com/torch/nngraph/blob/master/node.lua (This is not the split function for tensor)
    --local noise_i_split    = {noise_i:split(self.rhn_layers)} -- this nngraph.Node.split() function returns noutput number of new nodes that each take a single component of the output of this
    --local noise_h_split    = {noise_h:split(self.rhn_layers)} -- node in the order they are returned.
    --for layer_idx = 1, self.rhn_layers do     -- this self.rhn_layers is the vertical layer number of rhn, not the recurrent depth.
    --    local prev_h         = split[layer_idx]         -- the prev_h is the hidden state_s value from previous time step. Here it does not concern recurrent depth, which is sth studied inside rhn
    --    local n_i            = noise_i_split[layer_idx]     -- n_i and n_h are the dropout mask. n_i is the dropout mask for each (vertical) rhn layer's (vertical) input
    --    local n_h            = noise_h_split[layer_idx]     -- n_h is the dropout mask for each (horizontal) rhn unit.
    --    local next_h = self:buildRHNUnit(i[layer_idx - 1], prev_h, n_i, n_h)
    --    table.insert(next_s, next_h)
    --    i[layer_idx] = next_h   -- this next_h is the state_s value, which is the output of one rhn module (may contain multiple recurrent depth)
    --end
    --local dropped_o          = self:local_Dropout(i[self.rhn_layers], noise_o)   -- the output of the whole (probably multi-[vertical]layer) RHN module, after dropout
    --local module           = nn.gModule({x, prev_s, noise_i, noise_h, noise_o}, {dropped_o, nn.Identity()(next_s)})
    --module:getParameters():uniform(-0.04, 0.04)
    --return module
end

function BayesianLSTM:local_Dropout(input, noise)
    return nn.CMulTable()({input, noise})
end