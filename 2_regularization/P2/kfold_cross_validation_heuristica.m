function [ hypothesis, costes ] = kfold_cross_validation_heuristica(k,datos, ydatos, Grados, grados_max)
C = zeros([length(Grados) 2 grados_max]);
for grado_sel = 1:length(Grados)    % Para cada atributo
    best_errV=inf;
    best_i = 0;
    for i = 1:grados_max            % Se fija el resto de grados y se prueban distintos valores
        Grados(grado_sel) = i;

        err_T=0;
        err_V=0;
        datos_exp = expandir(datos, Grados);
        for fold = 1:k              %k-folds
            [Xcv, ycv, Xtr, ytr] = particion(fold,k,datos_exp,ydatos);
            [ Xtr_norm, mu, sig ] = normalizar(Xtr);
            w_norm = Xtr_norm\ytr;
            w = desnormalizar(w_norm, mu, sig);
            err_T = err_T + RMSE(w, Xtr, ytr);
            err_V = err_V + RMSE(w, Xcv, ycv);
        end
        %Calular media de errores
        err_T = err_T/k;
        err_V = err_V/k;
        %Para visualizacion
        C(grado_sel, 1, i) = err_T;
        C(grado_sel, 2, i) = err_V;

        if(err_V < best_errV)       % Guardar si es el mejor
            best_errV = err_V;
            best_i = i;
        end
    end
    Grados(grado_sel) = best_i;     %Fijar mejor valor para el atributo
end
hypothesis = Grados;
costes = C;
end

