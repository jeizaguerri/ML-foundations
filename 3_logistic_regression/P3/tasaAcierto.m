function [a] = tasaAcierto(ypred,y)
%Calculo de la tasa de acierto en función de las salidas predichas y reales
N = length(y);
a = (sum(ypred == y)) / N;
end

