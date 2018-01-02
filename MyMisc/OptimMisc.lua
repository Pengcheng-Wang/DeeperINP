---
--- Created by pwang8.
--- DateTime: 12/5/17 6:35 PM
--- This is implemented following Optim.lua script in OpenNMT lib
---
local OptimMisc = torch.class('OptimMisc')
--local class = require 'classic'
--local OptimMisc = classic.class('OptimMisc')

--[[ Clips gradients to a maximum L2-norm.

Parameters:

  * `gradParams` - a table of Tensor.
  * `maxNorm` - the maximum L2-norm.

]]
function OptimMisc.clipGradByNorm(gradParams, maxNorm)
    local gradNorm = 0
    if(torch.type(gradParams) == 'table') then
        for j=1, #gradParams do
            gradNorm = gradNorm + gradParams[j]:norm()^2
        end
    else
        gradNorm = gradParams:norm()^2
    end

    gradNorm = math.sqrt(gradNorm)
    -- Something added by pwang8
    if gradNorm == 0 then gradNorm = gradNorm + 1e-8 end

    local clipCoef = maxNorm / gradNorm

    if clipCoef < 1 then
        if(torch.type(gradParams) == 'table') then
            for j = 1, #gradParams do
                gradParams[j]:mul(clipCoef)
            end
        else
            gradParams:mul(clipCoef)
        end
    end
end

return OptimMisc