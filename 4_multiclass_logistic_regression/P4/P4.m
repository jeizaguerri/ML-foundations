%% Practica 4 Juan Eizaguerri Serrano

%% Carga de datos

clear ; close all;
%addpath(genpath('../minfunc'));

% Carga los datos y los permuta aleatoriamente

load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);

%Permutar datos
p = randperm(length(y));
X = X(p,:);
y = y(p);
p = randperm(length(ytest));
Xtest = Xtest(p,:);
ytest = ytest(p);

%Expandir
X_exp2 = expandir(X, ones(size(X,2),1)*2);
Xtest_exp2 = expandir(Xtest, ones(size(Xtest,2),1)*2);

X_exp3 = expandir(X, ones(size(X,2),1)*3);
Xtest_exp3 = expandir(Xtest, ones(size(Xtest,2),1)*3);

X_exp10 = expandir(X, ones(size(X,2),1)*10);
Xtest_exp10 = expandir(Xtest, ones(size(Xtest,2),1)*10);

nClases = 10;

%% 1. Regresión logística regularizada
fprintf('\n*** 1. Regresión logística regularizada ***\n');
%Modelo básico
fprintf('Modelo básico:\n');
%Entrenar
Thitas = entrenarLogReg(X, y, 0, nClases);
yPredTr = clasePredicha(X, Thitas);
ypredTest = clasePredicha(Xtest, Thitas);
%evaluar
c = confusionmat(y, yPredTr);
figure;
confusionchart(c);
title('Matriz de confusion para modelo básico');
taTr = tasaAcierto(yPredTr, y);
taTest = tasaAcierto(ypredTest, ytest);
fprintf('TaTr= %f     TaTest=%f\n', taTr, taTest);

%% Modelo con expansión grado 10 sin regularizar
fprintf('Modelo con expansión grado 10 sin regularizar:\n');
%Entrenar
Thitas = entrenarLogReg(X_exp10, y, 0, nClases);
yPred = clasePredicha(X_exp10, Thitas);
ypredTest = clasePredicha(Xtest_exp10, Thitas);
%Evaluar
c = confusionmat(y, yPred);
figure;
confusionchart(c);
title('Matriz de confusion para modelo grado 2 sin regularización\n');
taTr = tasaAcierto(yPred, y);
taTest = tasaAcierto(ypredTest, ytest);
fprintf('TaTr= %f     TaTest=%f\n', taTr, taTest);


%% Modelo con regularización. Elección de lambda por k-fold
fprintf('Modelo con regularización:\n');
%Elegir mejor lambda
Lambdas = 10.^(-15:5);
[bestLambda, Costes] = kfoldMultiReg(5, X, y, Lambdas, nClases);
fprintf('Mejor lambda encontrado: %d\n', bestLambda);
%Mostar evolución de la tasa de acierto en función de lambda
figure;
semilogx(Lambdas, Costes(:,1),'-b');
hold on;
semilogy(Lambdas, Costes(:,2), '-r');
grid on;
hold off;
xlabel('Lambda'); ylabel('Tasa acierto')
title('Evolución de la tasa de acierto en función de lambda')
legend('Entrenamiento', 'Validación');
%Entrenar con todos los datos
Thitas = entrenarLogReg(X, y, bestLambda, nClases);
yPred = clasePredicha(X, Thitas);
ypredTest = clasePredicha(Xtest, Thitas);
%Evaluar
c = confusionmat(y, yPred);
figure;
confusionchart(c);
title('Matriz de confusion para modelo grado 10 con regularización');
taTr = tasaAcierto(yPred, y);
taTest = tasaAcierto(ypredTest, ytest);
fprintf('TaTr= %f     TaTest=%f\n', taTr, taTest);


%% Impacto del grado de expansión en el resultado
fprintf('Elección del mejor grado de expansion usando mejor lambda:\n');
Grados = (1:20);
[TasasAciertos, bestGrados, taTr, taTest ]= probarGrados(X, y, Xtest, ytest, Grados, nClases);
%Mostrar evolución de tasas
figure;
plot(Grados, TasasAciertos(:,1), 'b-');
hold on;
plot(Grados, TasasAciertos(:,2), 'r-');
hold off;
grid on;
xlabel('Grados'); ylabel('Tasa de acierto');
legend('Datos entrenamiento', 'Datos test');
title('Tasa de acierto en función de grado de expansión');
fprintg('Mejor grado encontrado: %d\n', bestGrados);
fprintf('TaTr= %f     TaTest=%f\n', taTr, taTest);

%% 2. Evaluación del mejor modelo
fprintf('2. Evaluación del mejor modelo:\n');
grado = 3;
lambda = 10 ^-3;

Thitas = entrenarLogReg(X_exp3, y, lambda, nClases);

yPred = clasePredicha(X_exp3, Thitas);
yPredTest = clasePredicha(Xtest_exp3, Thitas);

%Tasa de acierto
taTr = tasaAcierto(yPred, y);
taTest = tasaAcierto(ypredTest, ytest);
fprintf('TaTr= %f     TaTest=%f\n', taTr, taTest);

%Matriz de confusión
verConfusiones(X, y, yPred);
c = confusionmat(y, yPred);
figure;
confusionchart(c);
title('Matriz de confusion para el mejor modelo (Train)');

verConfusiones(Xtest, ytest, yPredTest);
c = confusionmat(ytest, yPredTest);
figure;
confusionchart(c);
title('Matriz de confusion para el mejor modelo (Test)');

%Precisión recall para ver números problemáticos
for claseSel = (1:nClases)
    yClase = (y == claseSel);
    ytestClase = (ytest == claseSel);
    yPredClase = (yPred == claseSel);
    yPredTestClase = (yPredTest == claseSel);

    pTr = precision(yPredClase, yClase);
    pTest = precision(yPredTestClase, ytestClase);
    rTr = recall(yPredClase, yClase);
    rTest = recall(yPredTestClase, ytestClase);

    fprintf('Clase:%d:  pTr= %f      pTest=%f    rTr= %f     rTest=%f\n', claseSel, pTr, pTest, rTr, rTest);
    fprintf('Clasificados como %d: %d\n', claseSel, sum(yPredTestClase));
end


