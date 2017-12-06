---
--- Created by pwang8.
--- DateTime: 12/5/17 6:35 PM
--- This is implemented following Optim.lua script in OpenNMT lib
---

local OptimMisc = torch.class('OptimMisc')


--[[ Clips gradients to a maximum L2-norm.

Parameters:

  * `gradParams` - a table of Tensor.
  * `maxNorm` - the maximum L2-norm.

]]
function OptimMisc.clipGradByNorm(gradParams, maxNorm)
    local gradNorm = 0
    for j = 1, #gradParams do
        gradNorm = gradNorm + gradParams[j]:norm()^2
    end
    gradNorm = math.sqrt(gradNorm)

    local clipCoef = maxNorm / gradNorm

    if clipCoef < 1 then
        for j = 1, #gradParams do
            gradParams[j]:mul(clipCoef)
        end
    end
end