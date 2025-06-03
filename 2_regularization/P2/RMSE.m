function [error] = RMSE(tita, X, Y)
%Coste RMSE
N =length(Y);
error = sqrt(sum((valor_esperado(X,tita) - Y).^ 2) / N);
end

