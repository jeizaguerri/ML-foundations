function [a] = tasaError(ypred,y)
%Calculo de la tasa de error en función de las salidas predichas y reales
N = length(y);
a = (sum(ypred ~= y)) / N;
end

