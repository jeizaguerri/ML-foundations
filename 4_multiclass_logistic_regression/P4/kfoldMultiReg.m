function [bestLambda, TasasAcierto] = kfoldMultiReg(k, X, y, Lambdas, nClases)
%Busca el mejor lambda de los introducidos mediante kfold para el
%clasificador multiclase

TasasAcierto = zeros(size(y,2),2);
bestTasaAciertoV = 0;
i = 0;
for lambdaSel = Lambdas
    i = i+1;
    tasaAciertoT = 0;
    tasaAciertoV = 0;
    for fold = (1:k)
        %Particion de datos
        [ Xv, yv, Xtr, ytr ] = particion(fold, k, X, y);
        %Entrenar
        th = entrenarLogReg(Xtr, ytr, lambdaSel, nClases);
        %Evaluar
        tasaAciertoT = tasaAciertoT + tasaAcierto(clasePredicha(Xtr,th), ytr);
        tasaAciertoV = tasaAciertoV + tasaAcierto(clasePredicha(Xv,th), yv);
    end
    %Calcular medias
    tasaAciertoT = tasaAciertoT/k;
    tasaAciertoV = tasaAciertoV/k;
    %Almacenar
    TasasAcierto(i,1) = tasaAciertoT;
    TasasAcierto(i,2) = tasaAciertoV;
    %Guardar mejor
    if(tasaAciertoV > bestTasaAciertoV)
        bestTasaAciertoV = tasaAciertoV;
        bestLambda = lambdaSel;
    end
end
end

