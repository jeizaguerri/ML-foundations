function [TasasAcierto, mejorK, mejorTaVal, ErroresReproyeccion] = kfoldTaK(X, y, valoresK, lambdaReg, numClases)
% Calcula la tasa de acierto media para particiones de train y test con 
% distintos valores de k utilizando clasificador bayesiano de covarianzas
% completas

nPliegues = 5;
mejorK = 0;
mejorTaVal = 0;

TasasAcierto = zeros(2, size(valoresK, 2));
ErroresReproyeccion = zeros(2, size(valoresK, 2));

i = 0;
for k = valoresK
    i = i+1;
    taVal = 0;
    taTrain = 0;
    errRepVal = 0;
    errRepTrain = 0;
    for pliegue = (1:nPliegues)
        % Particion
        [ Xval, yval, Xtrain, ytrain ] = particion(pliegue, nPliegues, X, y );
        % Calculo PCA
        [U_ord, ~, mediaX] = pca(Xtrain);
        % Obtener Z
        Ztrain = transformarPca(Xtrain, mediaX, k, U_ord);
        Zval = transformarPca(Xval, mediaX, k, U_ord);
        % Entrenar modelo bayesiano
        modelo = entrenarGaussianas(Ztrain, ytrain, numClases, 0, lambdaReg);
        %Predicciones
        yhatTrain = clasificacionBayesiana(modelo, Ztrain);
        yhatVal = clasificacionBayesiana(modelo, Zval);
        %Evaluación
        taTrain = taTrain + tasaAcierto(yhatTrain, ytrain);
        taVal = taVal + tasaAcierto(yhatVal, yval);
        errRepTrain = errRepTrain + errorReproyeccion(Ztrain, Xtrain, U_ord, k, mediaX);
        errRepVal = errRepVal + errorReproyeccion(Zval, Xval, U_ord, k, mediaX);
        
    end
    %Calcular medias
    taTrain = taTrain / nPliegues ;
    taVal = taVal / nPliegues ;
    errRepTrain = errRepTrain /nPliegues;
    errRepVal = errRepVal /nPliegues;
    %Guardar
    TasasAcierto(1, i) = taTrain;
    TasasAcierto(2, i) = taVal;
    ErroresReproyeccion(1, i) = errRepTrain;
    ErroresReproyeccion(2, i) = errRepVal;
    %Guardar si es el mejor
    if taVal > mejorTaVal
        mejorTaVal = taVal;
        mejorK = k;
    end
end

end

