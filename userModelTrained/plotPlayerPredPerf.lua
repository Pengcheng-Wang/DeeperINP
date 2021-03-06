---
--- Created by pwang8.
--- DateTime: 11/3/17 1:25 PM
---

local gnuplot = require 'gnuplot'

for _randSeed=1, 1 do
    local _resDic = string.format('2L64_48_lstm/rnndrop.15/seed%d', _randSeed)
    for _testSeed=1,5 do
        gnuplot.pngfigure(paths.concat(_resDic, string.format('actPA%d.png', _testSeed)))
        -- read test-set data from file
        local _accFile = io.open(paths.concat(_resDic, string.format('no_aug/tdiv%d', _testSeed), 'test.log'), 'r')
        local _actAccNoAug = {}
        local _ = _accFile:read() -- This is the header in the file, read and ignore this line
        for _line in _accFile:lines('*l') do
            local _oneLine = _line:split('\t')
            if #_oneLine == 3 then
                _actAccNoAug[#_actAccNoAug + 1] = tonumber(string.sub(_oneLine[2], 1, -2))/100.0 -- the 2nd item in each line is the act prediction accuracy, and need to take out the % char at the end
            end
        end
        -- read training-set data from file
        local _accFile = io.open(paths.concat(_resDic, string.format('no_aug/tdiv%d', _testSeed), 'train.log'), 'r')
        local _actAccNoAugTrain = {}
        local _ = _accFile:read() -- This is the header in the file, read and ignore this line
        for _line in _accFile:lines('*l') do
            _actAccNoAugTrain[#_actAccNoAugTrain + 1] = tonumber(_line)/100.0 -- only one item in train.log in each row
        end
        -- now read the test-set file from the aug directory
        _accFile = io.open(paths.concat(_resDic, string.format('aug/tdiv%d', _testSeed), 'test.log'), 'r')
        local _actAccAug = {}
        local _ = _accFile:read()   -- Pass the first row/line in the file, which contains header
        for _line in _accFile:lines('*l') do
            local _oneLine = _line:split('\t')
            if #_oneLine == 3 then
                _actAccAug[#_actAccAug + 1] = tonumber(string.sub(_oneLine[2], 1, -2))/100.0 -- the 2nd item in each line is the act prediction accuracy, and need to take out the % char at the end
            end
        end
        -- now read the training-set file from the aug directory
        _accFile = io.open(paths.concat(_resDic, string.format('aug/tdiv%d', _testSeed), 'train.log'), 'r')
        local _actAccAugTrain = {}
        local _ = _accFile:read()   -- Pass the first row/line in the file, which contains header
        for _line in _accFile:lines('*l') do
            _actAccAugTrain[#_actAccAugTrain + 1] = tonumber(_line)/100.0 -- only one acc in train.log
        end
        -- Can draw the figure using the two tables, need to construct tensors based on the two tables
        gnuplot.plot({'no aug test', torch.Tensor(_actAccNoAug), '-'}, {'aug test', torch.Tensor(_actAccAug), '-'}, {'no aug train', torch.Tensor(_actAccNoAugTrain), '-'}, {'aug train', torch.Tensor(_actAccAugTrain), '-'})
        gnuplot.raw('set xtics <INCREMENT>')
        gnuplot.raw('set term png size 1000, 400')
        gnuplot.raw('set yr [GPVAL_DATA_Y_MIN:GPVAL_DATA_Y_MAX]')
        --gnuplot.raw('set size 1.5,2')
        gnuplot.plotflush()
        gnuplot.axis({1, math.max(#_actAccNoAug, #_actAccAug), 0, ''})
        gnuplot.title(string.format('Act Pred Accu when testSeed is %d', _testSeed))

        gnuplot.pngfigure(paths.concat(_resDic, string.format('actPA%dTestOnly.png', _testSeed)))
        gnuplot.plot({'no aug test', torch.Tensor(_actAccNoAug), '-'}, {'aug test', torch.Tensor(_actAccAug), '-'})
        gnuplot.raw('set xtics <INCREMENT>')
        gnuplot.raw('set term png size 1000, 400')
        -- gnuplot.raw('set yr [GPVAL_DATA_Y_MIN:GPVAL_DATA_Y_MAX]')
        gnuplot.raw('set yrange [0.2:0.45]')
        --gnuplot.raw('set size 1.5,2')
        gnuplot.plotflush()
        gnuplot.axis({1, math.max(#_actAccNoAug, #_actAccAug), 0, ''})
        gnuplot.title(string.format('Act Pred Accu when testSeed is %d', _testSeed))
    end
    gnuplot.close()
end
