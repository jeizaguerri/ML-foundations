function [r] = recall(yPred, yReal)
tp = sum(yPred & yReal);
fn = sum(~yPred & yReal);
r = tp / (tp + fn);
end
