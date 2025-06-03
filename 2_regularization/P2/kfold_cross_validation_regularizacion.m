function [ hypothesis, costes ] = kfold_cross_validation_regularizacion(k,datos, ydatos, Grados, Lambdas)
C = zeros([2 length(Lambdas)]);
best_errV=inf;
best_lambda = 0;
i = 0;
for lambda_sel = Lambdas            % Se fija el resto de grados y se prueban distintos valores
    i = i+1;
    err_T=0;
    err_V=0;
    datos_exp = expandir(datos, Grados);
    for fold = 1:k              %k-folds
        [Xcv, ycv, Xtr, ytr] = particion(fold,k,datos_exp,ydatos);
        [ Xtr_norm, mu, sig ] = normalizar(Xtr);

        D = sum(Grados(:));
        H = Xtr_norm' * Xtr_norm + lambda_sel * diag([0 ones(1,D)]);
        w_norm = H \ (Xtr_norm' * ytr);
        w = desnormalizar(w_norm, mu, sig);

        err_T = err_T + RMSE(w, Xtr, ytr);
        err_V = err_V + RMSE(w, Xcv, ycv);
    end
    %Calular media de errores
    err_T = err_T/k;
    err_V = err_V/k;
    %Para visualizacion
    C(1, i) = err_T;
    C(2, i) = err_V;

    if(err_V < best_errV)       % Guardar si es el mejor
        best_errV = err_V;
        best_lambda = lambda_sel;
    end
end
hypothesis = best_lambda;
costes = C;
end

