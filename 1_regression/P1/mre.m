function [error] = mre(X,Y,tita,N)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
error = sum(abs(valor_esperado(X,tita) - Y) ./ Y) / N;
end