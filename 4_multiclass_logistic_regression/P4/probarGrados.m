function [TasasAcierto, bestGrado, bestTaTrain, bestTaTest] = probarGrados(X, y, Xtest, ytest, Grados, nClases)
TasasAcierto = zeros(size(Grados,2), 2);
lambda = 10 ^ -3;
i = 0;
bestTaTest = 0;
for grado_sel = Grados
    i = i+1;
    %Entrenar con todos los datos
    X_exp= expandir(X, ones(size(X,2),1)*grado_sel);
    Xtest_exp = expandir(Xtest, ones(size(X,2),1)*grado_sel);

    Thitas = entrenarLogReg(X_exp, y, lambda, nClases);
    yPred = clasePredicha(X_exp, Thitas);
    ypredTest = clasePredicha(Xtest_exp, Thitas);
    %Evaluar
    taTr = tasaAcierto(yPred, y);
    taTest = tasaAcierto(ypredTest, ytest);
    TasasAcierto(i,1) = taTr;
    TasasAcierto(i,2) = taTest;
    %Guardar el mejor
    if(taTest > bestTaTest)
        bestGrado = grado_sel;
        bestTaTrain = taTr;
        bestTaTest  = taTest;
    end
end
end

