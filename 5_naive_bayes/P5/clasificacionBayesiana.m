function yhat = clasificacionBayesiana( modelo, X)
% Con los modelos entrenados, predice la clase para cada muestra X
% Inicialización de variables
nc = length(modelo);
n = size(X, 1);
probs = zeros(n, nc);
    
% Cálculo de la probabilidad logarítmica para cada clase
for i = 1:nc
    % Vector de medias de la clase i
    mu = modelo(i).mu;
    % Matriz de covarianzas de la clase i
    Sigma = modelo(i).Sigma;
    % Cálculo de la probabilidad logarítmica de la clase i para cada muestra
    probs(:,i) = gaussLog(mu, Sigma, X);
end
    
    % Clasificación de cada muestra
    [~, yhat] = max(probs, [], 2);

