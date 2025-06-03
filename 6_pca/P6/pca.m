function [U_ord, ValoresPropios_ord, mediaX] = pca(X)
% Calcula los valores y vectores propios ordenados para un conjunto de
% datos de entrenamiento X

% Estandarizar medias
mediaX = mean(X);
X_est = X - mediaX;
% Calcular matriz de covarianzas
covarianzasX = cov(X_est);
% Obtener valores y vectores propios de la matariz de covarianzas
[U, A] = eig(covarianzasX);
ValoresPropios = diag(A);
% Ordenar los vectores de U en funcion de los valores de A
[ValoresPropios_ord, indices] = sort(ValoresPropios, 'descend');
U_ord = U(:,indices);

end

