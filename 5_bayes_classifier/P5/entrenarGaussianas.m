function modelo = entrenarGaussianas( Xtr, ytr, nc, NaiveBayes, landa )
% Entrena una Gaussana para cada clase y devuelve:
% modelo{i}.N     : Numero de muestras de la clase i
% modelo{i}.mu    : Media de la clase i
% modelo{i}.Sigma : Covarianza de la clase i
% Si NaiveBayes = 1, las matrices de Covarianza serán diagonales
% Se regularizarán las covarianzas mediante: Sigma = Sigma + landa*eye(D)

% Inicialización del modelo
modelo = struct();
for i = 1:nc
    Xi = Xtr(ytr == i, :);
    modelo(i).N = sum(ytr==i); % Numero de muestras de la clase i
    modelo(i).mu = mean(Xi); % Media de la clase i
    modelo(i).Sigma = cov(Xi); % Covarianza de la clase i
    % Regularización de la covarianza
    modelo(i).Sigma = modelo(i).Sigma + landa*eye(size(Xtr,2));
    % Si NaiveBayes = 1, las matrices de Covarianza serán diagonales
    if NaiveBayes
        modelo(i).Sigma = diag(diag(modelo(i).Sigma));
    end
end