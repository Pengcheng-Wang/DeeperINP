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

function TableSetMisc.static.fastLSTMForgetGateInit(fLstmM, dropoutRate, fLstmSize, nninit)
    assert(type(fLstmM.i2g) == 'table')
    if dropoutRate > 0 then
        fLstmM.i2g.modules[2].modules[3]:init('bias', nninit.constant, 1)
    else
        fLstmM.i2g:init({'bias', {{2*fLstmSize+1, 3*fLstmSize}}}, nninit.constant, 1)
    end
end

return TableSetMisc