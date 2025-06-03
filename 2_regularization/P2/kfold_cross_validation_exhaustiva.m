function [ best_hypothesis, best_err ] = kfold_cross_validation_exhaustiva(k,datos, ydatos, Grados, grados_max, nivel_rec)
%Caso base    
if nivel_rec == length(Grados)
    best_err=inf;
    for i = 1:grados_max
        Grados(nivel_rec) = i;
    
        err_V = 0;
        datos_exp = expandir(datos, Grados);
        for fold = 1:k              %k-folds
            [Xcv, ycv, Xtr, ytr] = particion(fold,k,datos_exp,ydatos);
            [ Xtr_norm, mu, sig ] = normalizar(Xtr);
            w_norm = Xtr_norm\ytr;
            w = desnormalizar(w_norm, mu, sig);
            err_V = err_V + RMSE(w, Xcv, ycv);
        end
        %Calular media de errores
        err_V = err_V/k;

        if(err_V < best_err)       % Guardar si es el mejor
            best_err = err_V;
            best_hypothesis = Grados;
        end
    end
    return;
end

%Caso recursivo
best_err = inf;
for i = 1:grados_max
    Grados(nivel_rec) = i;
    [best_hypothesis_rec, best_err_rec] = kfold_cross_validation_exhaustiva(k, datos, ydatos, Grados, grados_max, nivel_rec+1);
    if best_err_rec < best_err
        best_err = best_err_rec;
        best_hypothesis = best_hypothesis_rec;
    end
end

end
