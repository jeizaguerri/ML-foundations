function [Zk] = transformarPca(X, mediaX, k, U_ord)
% Aplica la transformacion Z=X*U para los primeros k vectores de U.

% Estandarizar medias
X_est = X - mediaX;

% Transformar
Zk = X_est * U_ord(:, 1:k);
end

