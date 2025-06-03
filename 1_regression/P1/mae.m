function [error] = mae(X,Y,tita,N)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
error = sum(abs(valor_esperado(X,tita) - Y)) / N;
end