-- Creates aggregator module for a dueling architecture based on a number of discrete actions
-- Comments by pwang8. Input into the DuelAggregator module is {value_stream, advantage_stream}
-- value_stream is the state value estimation, which is only one value for one state input (for each s-a-s data point)
-- advantage_stream is the advantage values estimation, which contains m values with m being the # of actions
-- param m of this function is # of actions
local DuelAggregator = function(m)
  local aggregator = nn.Sequential()
  local aggParallel = nn.ParallelTable()
  
  -- Advantage duplicator (for calculating and subtracting mean)
  local advDuplicator = nn.Sequential()
  local advConcat = nn.ConcatTable()
  advConcat:add(nn.Identity())
  -- Advantage mean duplicator
  local advMeanDuplicator = nn.Sequential()
  advMeanDuplicator:add(nn.Mean(1, 1))  -- nn.Mean returns the mean along 2nd dim here (1st dim treated as batch index).
  advMeanDuplicator:add(nn.Replicate(m, 2, 2))  -- replicate the input for m times, replication is along 2nd index (as action index). Not sure about the exact meaning. Anyway, Output of this duplication is in 3-dim. 1st dim is batch index, 2nd dim is action index, 3rd dim is just 1.
  advConcat:add(advMeanDuplicator)
  advDuplicator:add(advConcat)
  -- Subtract mean from advantage values
  advDuplicator:add(nn.CSubTable())   -- subtract adv.mean() from adv. This type of advantage values processing is adopted for stablization purpose
  
  -- Add value and advantage duplicators
  aggParallel:add(nn.Replicate(m, 2, 2))  -- duplicate the state values for m times
  aggParallel:add(advDuplicator)    -- this is the (adv - adv.mean())

  -- Calculate Q^ = V^ + A^
  aggregator:add(aggParallel)
  aggregator:add(nn.CAddTable())

  return aggregator
end

return DuelAggregator
