function [p] = precision(yPred, yReal)
tp = sum(yPred & yReal);
fp = sum(yPred & ~yReal);
if tp == 0 && fp==0
    p = 1;
else
    p = tp / (tp + fp);
end
end

