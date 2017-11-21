--
-- User: pwang8
-- Date: 1/21/17
-- Time: 8:29 PM
-- Places to put setContains function
--
local class = require 'classic'
local TableSetMisc = classic.class('TableSetMisc')

function TableSetMisc.static.addToSet(set, key)
    set[key] = true
end

function TableSetMisc.static.removeFromSet(set, key)
    set[key] = nil
end

function TableSetMisc.static.setContains(set, key)
    return set[key] ~= nil
end

function TableSetMisc.static.countsInSet(set)
    local cnt = 0
    for k,v in pairs(set) do
        cnt = cnt + 1
    end
    return cnt
end

function TableSetMisc.static.tableContainsValue(tab, element)
    for _, value in pairs(tab) do
        if value == element then
            return true
        end
    end
    return false
end

--- This function is designed to do forget gate bias initialization
--- FastLSTM updated its implementation, so it contains the "lazy" dropout when dropout
--- rate is not 0
--- Params: fLstmSize: number of hidden nuerons in LSTM
--- nninit: the reference to the nninit object
function TableSetMisc.static.fastLSTMForgetGateInit(fLstmM, dropoutRate, fLstmSize, nninit)
    assert(type(fLstmM.i2g) == 'table')
    if dropoutRate > 0 then
        -- If dropoutRate is > 0, then the lazy droput is adopted in FastLSTM. So the
        -- structure of the NN looks like the following
        -- -- Oct 6, 2017. I updated the ElementResearch.rnn lib
        -- They've made changes, which makes this lstm forget gate bias init code not working
        -- I checked the code. The nninit has not been changed. It is solely the changes from lstm
        -- implementation makes the problem. The current FastLSTM implementation included a "soft dropout"
        -- module, which implements Gal's Bayesian RNN. So, FastLSTM.i2g right now has multiple tables in it.
        -- So the following way demonstates how to get access to the forget gate, and then do initialization
        -- And also here's the structure of FastLSTM.i2g
        --nn.Sequential {
        --    [input -> (1) -> (2) -> (3) -> output]
        --    (1): nn.ConcatTable {
        --        input
        --        |`-> (1): nn.Dropout(0.2, lazy)
        --        |`-> (2): nn.Dropout(0.2, lazy)
        --        |`-> (3): nn.Dropout(0.2, lazy)
        --        `-> (4): nn.Dropout(0.2, lazy)
        --        ... -> output
        --    }
        --    (2): nn.ParallelTable {
        --        input
        --        |`-> (1): nn.Linear(21 -> 32)
        --        |`-> (2): nn.Linear(21 -> 32)
        --        |`-> (3): nn.Linear(21 -> 32)
        --        `-> (4): nn.Linear(21 -> 32)
        --        ... -> output
        --    }
        --    (3): nn.JoinTable
        --}
        -- We assume input layer to this LSTM has 21 neurons,
        -- the number of hidden nuerons in the LSTM is 32
        fLstmM.i2g.modules[2].modules[3]:init('bias', nninit.constant, 1)
    else
        -- If dropoutRate == 0, meaning dropOut is not applied, then
        -- the structure of i2g is a nn.Linear(21->128), which merges 4 gates in one
        fLstmM.i2g:init({'bias', {{2*fLstmSize+1, 3*fLstmSize}}}, nninit.constant, 1)
        --fLstmM.i2g:init({'bias', {{3*fLstmSize+1, 4*fLstmSize}}}, nninit.constant, 1)
    end
end

--- This function is used by rnn modules which utilize dropout masks from outside of the modules, e.g., rhn, BayesianLSTM
--- An assumption here is for multi-layer RNN modules, each hidden layer has the same size (number of hidden neurons)
--- params:
--- rnn_noise_i, rnn_noise_h, rnn_noise_o are refs to tables of dropout mask tensor for rnn models
--- gateNum is number of gates adopted by the model. RHN has 2 gates, LSTM has 4 gates.
function TableSetMisc.static.buildRNNDropoutMask(rnn_noise_i, rnn_noise_h, rnn_noise_o, _inSize, _outSize, _hiddenLayerCnt, _batchSize, lstmHistLen, gateNum)
    -- rnn_noise_i, rnn_noise_h, rnn_noise_o are dropout masks for input, hidden, and output neurons for rnn models
    rnn_noise_i[1] = {}
    rnn_noise_h[1] = {}
    rnn_noise_o[1] = torch.zeros(_batchSize, _outSize)
    for _d = 1, _hiddenLayerCnt do
        if _d == 1 then
            rnn_noise_i[1][_d] = torch.zeros(_batchSize, gateNum * _inSize)
        else
            rnn_noise_i[1][_d] = torch.zeros(_batchSize, gateNum * _outSize)  -- assumption: rnn/rhn/lstm with more than 2 hidden layers would have same size for each hidden layer
        end
        rnn_noise_h[1][_d] = torch.zeros(_batchSize, gateNum * _outSize)
    end

    for _h=2, lstmHistLen do
        rnn_noise_o[_h] = rnn_noise_o[1]:clone()
        rnn_noise_i[_h] = {}
        rnn_noise_h[_h] = {}
        for _d = 1, _hiddenLayerCnt do
            rnn_noise_i[_h][_d] = rnn_noise_i[1][_d]:clone()
            rnn_noise_h[_h][_d] = rnn_noise_h[1][_d]:clone()
        end
    end
end

--- This function is used by rnn modules which utilize dropout masks from outside of the modules, e.g., rhn, BayesianLSTM
--- An assumption here is for multi-layer RNN modules, each hidden layer has the same size (number of hidden neurons)
function TableSetMisc.static.sampleRNNDropoutMask(prob, rnn_noise_i, rnn_noise_h, rnn_noise_o, _hiddenLayerCnt, lstmHistLen)
    assert(prob>=0 and prob<1, 'Dropout prob should be in [0,1)')
    if prob>0 then
        rnn_noise_o[1]:bernoulli(1 - prob)
        rnn_noise_o[1]:div(1 - prob)
        for _d = 1, _hiddenLayerCnt do
            rnn_noise_i[1][_d]:bernoulli(1 - prob)
            rnn_noise_i[1][_d]:div(1 - prob)
            rnn_noise_h[1][_d]:bernoulli(1 - prob)
            rnn_noise_h[1][_d]:div(1 - prob)
        end
        for _h=2, lstmHistLen do
            rnn_noise_o[_h] = rnn_noise_o[1]:clone()
            for _d = 1, _hiddenLayerCnt do
                rnn_noise_i[_h][_d] = rnn_noise_i[1][_d]:clone()
                rnn_noise_h[_h][_d] = rnn_noise_h[1][_d]:clone()
            end
        end
    else    -- prob == 0
        for _h=1, lstmHistLen do
            rnn_noise_o[_h]:zero():add(1)
            for _d = 1, _hiddenLayerCnt do
                rnn_noise_i[_h][_d]:zero():add(1)
                rnn_noise_h[_h][_d]:zero():add(1)
            end
        end
    end
end

return TableSetMisc