function [bestLambda, C] = kfoldLogReg(k, X, y, Lambdas)
%k-fod cross-validation para elegir el mejor lambda para una regresión logística
%Los datos de entrada tienen deben estar ya expandidos

options.useMex = 1 ;
options.display = 'none';
options.method = 'newton';

C = zeros(2, length(Lambdas));
i= 0;
bestTasaV = 0;

for lambdaSel = Lambdas
    tasaV = 0;
    tasaT = 0;
    %k folds
    for fold = (1:k)
        %particion de datos
        [Xv, yv, Xt, yt] = particion(fold, k, X, y);

        %Entrenar modelo
        th_ini = zeros(1, size(X,2))';
        th = minFunc(@CosteLogReg, th_ini , options, Xt, yt, lambdaSel);

        %Evaluar
        tasaV = tasaV + tasaAcierto(prediccion(Xv * th, 0.5),yv);
        tasaT = tasaT + tasaAcierto(prediccion(Xt * th, 0.5),yt);
    end
    %Calcular medias de costes y almacenar en el vector
    tasaV = tasaV / k;
    tasaT = tasaT /k;
    i = i+1;
    C(1, i) = tasaT;
    C(2, i) = tasaV;

    if(tasaV > bestTasaV)
        bestTasaV = tasaV;
        bestLambda = lambdaSel;
    end
end