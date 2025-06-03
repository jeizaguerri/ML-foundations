function [bestLambda, TasasAcierto] = elegirLambdaKfold(X, y, numClases, ingenuo, Lambdas, k)
%Selección del mejor parámetro de regularización lambda por k-fold
%para clasificación con modelo gaussiano

i = 0;
bestTaV = 0;
TasasAcierto = zeros(2, size(Lambdas, 2));
for lambdaSel = Lambdas
    i = i+1;
    taTr = 0;
    taV = 0;
    for fold = (1:k)
        %Particion
        [ Xcv, ycv, Xtr, ytr ] = particion( fold, k, X, y );
        %Entrenamiento con datos de entrenamiento
        modelo = entrenarGaussianas(Xtr, ytr, numClases, ingenuo, lambdaSel);
        %Predicciones
        yhatTr = clasificacionBayesiana(modelo, Xtr);
        yhatV = clasificacionBayesiana(modelo, Xcv);
        %Evaluación
        taTr = taTr + tasaAcierto(yhatTr, ytr);
        taV = taV + tasaAcierto(yhatV, ycv);
    end
    %Calcular medias
    taTr = taTr / k;
    taV = taV / k;
    %Guardar
    TasasAcierto(1, i) = taTr;
    TasasAcierto(2, i) = taV;
    %Guardar si es mejor
    if (taV > bestTaV)
        bestTaV = taV;
        bestLambda = lambdaSel;
    end
end
end

