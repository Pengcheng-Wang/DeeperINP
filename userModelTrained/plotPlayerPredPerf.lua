---
--- Created by pwang8.
--- DateTime: 11/3/17 1:25 PM
---

local gnuplot = require 'gnuplot'

local _resDic = 'seed1'
gnuplot.pngfigure(paths.concat(_resDic, 'actPA.png'))
for _testSeed=1,5 do
    -- read data from file
    local _accFile = io.open(paths.concat(_resDic, string.format('no_aug/tdiv%d', _testSeed), 'test.log'), 'r')
    local _actAccNoAug = {}
    local _ = _accFile:read() -- This is the header in the file, read and ignore this line
    for _line in _accFile:lines('*l') do
        local _oneLine = _line:split('|')
        if #_oneLine == 3 then
            _actAccNoAug[#_actAccNoAug + 1] = tonumber(string.sub(_oneLine[2], 1, -2)) -- the 2nd item in each line is the act prediction accuracy, and need to take out the % char at the end
        end
    end
    -- now read the file from the aug directory
    _accFile = io.open(paths.concat(_resDic, string.format('aug/tdiv%d', _testSeed), 'test.log'), 'r')
    local _actAccAug = {}
    local _ = _accFile:read()   -- Pass the first row/line in the file, which contains header
    for _line in _accFile:lines('*l') do
        local _oneLine = _line:split('|')
        if #_oneLine == 3 then
            _actAccAug[#_actAccAug + 1] = tonumber(string.sub(_oneLine[2], 1, -2)) -- the 2nd item in each line is the act prediction accuracy, and need to take out the % char at the end
        end
    end
    -- Can draw the figure using the two tables, need to construct tensors based on the two tables
    gnuplot.plot({'no_aug', torch.Tensor(_actAccNoAug)}, {'aug', torch.Tensor(_actAccAug)})
    gnuplot.plotflush()
    gnuplot.axis({1, math.max(#_actAccNoAug, #_actAccAug), 0, ''})
    gnuplot.title(string.format('Act Pred Accu when testSeed is %d', _testSeed))
end
