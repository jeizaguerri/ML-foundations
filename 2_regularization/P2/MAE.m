function [error] = MAE(tita, X, Y)
%Coste MAE
N =length(Y);
error = sum(abs(valor_esperado(X,tita) - Y)) / N;
end

