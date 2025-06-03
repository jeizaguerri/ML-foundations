%% P3: Cargar y mostrar datos de clasificación de vinos

clear ; close all;

% Optiones para minFunc
options.useMex = 1 ;
options.display = 'none';
options.method = 'newton';

% Cargar y preparar datos
wine_train = load('wine_train.txt');
wine_test  = load('wine_test.txt');

clase = 1;
f1 = 6; f2 = 10; 

ydata = wine_train(:,1);
Xdata = wine_train(:,2:end);
x1 = Xdata(:,f1);
x2 = Xdata(:,f2);
y = ydata==clase;
X = [ones(length(y),1) x1 x2];

yt = wine_test(:,1);
Xt = wine_test(:,2:end);
xt1 = Xt(:,f1);
xt2 = Xt(:,f2);
ytest = yt==clase;
Xtest = [ones(length(ytest),1) xt1 xt2];

%% 1. Regresión logística básica 
fprintf('\n*** 1. Regresión logística básica  ***\n');

%Regresion basica
th_ini = ones(1,3)';
theta = minFunc(@CosteLogistico, th_ini , options, X, y);

%Dibujar superficie de separación
plot_wines(Xdata, ydata, f1, f2);
plotDecisionBoundary(theta, X);
title('Regresión logística básica');

%Evaluación
hTrBasica = X * theta;
pred = prediccion(hTrBasica, 0.5);
taTr = tasaAcierto(pred, y);
teTr = tasaError(pred, y);
fprintf('Regresión básica:\n');
fprintf('Datos de entrenamiento: Ta=%f; Te=%f\n', taTr, teTr);

hTestBasica = Xtest * theta;
pred = prediccion(hTestBasica, 0.5);
taTest = tasaAcierto(pred, ytest);
teTest = tasaError(pred, ytest);
fprintf('Datos de test: Ta=%f; Te=%f\n', taTest, teTest);


%Con expansion
grado = 2;
X_exp = expandir2(x1, x2, grado);
th_ini = ones(1,size(X_exp,2))' .* 0;
theta_exp = minFunc(@CosteLogistico, th_ini , options, X_exp, y);

plot_wines(Xdata, ydata, f1, f2);
plotDecisionBoundary(theta_exp, X);
title('Regresión logística con expansión grado 2');

%Evaluación
XTest_exp = expandir2(xt1, xt2, grado);

hTrGrado2 = X_exp * theta_exp;
pred = prediccion(hTrGrado2, 0.5);
taTr = tasaAcierto(pred, y);
teTr = tasaError(pred, y);
fprintf('Expansión grado 2:\n');
fprintf('Datos de entrenamiento: Ta=%f; Te=%f\n', taTr, teTr);

hTestGrado2 = XTest_exp * theta_exp;
pred = prediccion(hTestGrado2, 0.5);
taTest = tasaAcierto(pred, ytest);
teTest = tasaError(pred, ytest);
fprintf('Datos de test: Ta=%f; Te=%f\n', taTest, teTest);


%Fijar atributo 6 a 0.6, gráfica de probabilidades en función
%del atributo 10.
valorFijo6 = 0.6;
valores10 = 0.1 .* (0:10);
Xpuntos = [ones(1,11); ones(1,11) * valorFijo6; valores10]';
H = sigmoidal(Xpuntos * theta);

%Mostrar grafica
figure;
plot(valores10, H, 'b-');
grid on;
title(sprintf('Probabilidad de clase %d en función de atributo %d', clase, f2));
xlabel(sprintf('Atributo %d', f2));
ylabel(sprintf('Probabilidad de clase %d', clase));

%% 2. Regularización
fprintf('\n*** 2. Regularización ***\n');

%Prueba de expansión sin regularización para ver sobreajuste
%Con expansion
grado = 6;
X_exp = expandir2(x1, x2, grado);
th_ini = ones(1,size(X_exp,2))' .* 0;
theta_exp = minFunc(@CosteLogistico, th_ini , options, X_exp, y);

plot_wines(Xdata, ydata, f1, f2);
plotDecisionBoundary(theta_exp, X);
title('Regresión logística con expansión grado 6');

%Evaluación
XTest_exp = expandir2(xt1, xt2, grado);

hTr = X_exp * theta_exp;
pred = prediccion(hTr, 0.5);
taTr = tasaAcierto(pred, y);
teTr = tasaError(pred, y);
fprintf('Expansión sin regularización para comprobar sobreajuste:\n');
fprintf('Datos de entrenamiento: Ta=%f; Te=%f\n', taTr, teTr);

hTest = XTest_exp * theta_exp;
pred = prediccion(hTest, 0.5);
taTest = tasaAcierto(pred, ytest);
teTest = tasaError(pred, ytest);
fprintf('Datos de test: Ta=%f; Te=%f\n', taTest, teTest);


%Elegir parámetro de regularización lambda por k-fold
grados = 6;
X_exp = expandir2(x1,x2,grados);
Lambdas = 10.^(-10:3);

[best_lambda, C] = kfoldLogReg(5, X_exp, y, Lambdas);

%Mostrar evolución de la tasa de error
figure;
semilogx(Lambdas, C(1,:), '-r');
hold on;
semilogx(Lambdas, C(2,:), '-b');
hold off;
grid on;
title('Evolución de la tasa de acierto en función de lambda');
xlabel('Lambda'); ylabel('Tasa de acierto');
legend('Datos entrenamiento', 'Datos validación')

%Entrenar modelo con todos los datos de entrenamiento
th_ini = ones(1,size(X_exp,2))' .* 0;
theta_exp = minFunc(@CosteLogReg, th_ini , options, X_exp, y, best_lambda);

XTest_exp = expandir2(xt1, xt2, grado);

%Evaluación
hTrReg = X_exp * theta_exp;
pred = prediccion(hTrReg, 0.5);
taTr = tasaAcierto(pred, y);
teTr = tasaError(pred, y);
fprintf('Expansión con regularización lambda = :%f\n', best_lambda);
fprintf('Datos de entrenamiento: Ta=%f; Te=%f\n', taTr, teTr);

hTestReg = XTest_exp * theta_exp;
pred = prediccion(hTestReg, 0.5);
taTest = tasaAcierto(pred, ytest);
teTest = tasaError(pred, ytest);
fprintf('Datos de test: Ta=%f; Te=%f\n', taTest, teTest);

%Mostar modelo
plot_wines(Xdata, ydata, f1, f2);
plotDecisionBoundary(theta_exp, X);
title(sprintf('Regresión logística regularizada grado %d, lambda=%f', grados, best_lambda));

%Para lambda = 0
%Entrenar modelo con todos los datos de entrenamiento
th_ini = ones(1,size(X_exp,2))' .* 0;
theta_expLambda0 = minFunc(@CosteLogReg, th_ini , options, X_exp, y, 0);

XTest_exp = expandir2(xt1, xt2, grado);

%Evaluación
hTrGrado6mala = X_exp * theta_expLambda0;
pred = prediccion(hTrGrado6mala, 0.5);
taTr = tasaAcierto(pred, y);
teTr = tasaError(pred, y);
fprintf('Expansión con regularización lambda = 0:\n');
fprintf('Datos de entrenamiento: Ta=%f; Te=%f\n', taTr, teTr);

hTestGrado6mala = XTest_exp * theta_expLambda0;
pred = prediccion(hTestGrado6mala, 0.5);
taTest = tasaAcierto(pred, ytest);
teTest = tasaError(pred, ytest);
fprintf('Datos de test: Ta=%f; Te=%f\n', taTest, teTest);

%Mostar modelo
plot_wines(Xdata, ydata, f1, f2);
plotDecisionBoundary(theta_expLambda0, X);
title(sprintf('Regresión logística regularizada grado %d, lambda=%f', grados, 0));


%Fijar atributo 6 a 0.6, gráfica de probabilidades en función
%del atributo 10.
valorFijo6 = 0.6;
valores10 = 0.1 .* (0:10);
Xpuntos_exp = expandir2((ones(1,11) * valorFijo6)', valores10', grados);
H = sigmoidal(Xpuntos_exp * theta_exp);

%Mostrar grafica
figure;
plot(valores10, H, 'b-');
grid on;
title(sprintf('Probabilidad de clase %d en función de atributo %d con regularización', clase, f2));
xlabel(sprintf('Atributo %d', f2));
ylabel(sprintf('Probabilidad de clase %d', clase));


%% 3. Curvas Precisión/Recall
fprintf('\n*** 3. Curvas Precisión/Recall ***\n');

%Plot de curvas precision/recall y calculo de la mejor precisión para
%recall >= 0.9
bestPBasica = plotPrecisionRecall(hTestBasica, ytest);
title('Curva Precisión/Recall para regresión básica')

bestPGrado2 = plotPrecisionRecall(hTestGrado2, ytest);
title('Curva Precisión/Recall para regresión grado 2')

bestPReg = plotPrecisionRecall(hTestReg, ytest);
title('Curva Precisión/Recall para regresión regularizada grado 6')

bestPGrado6Mala = plotPrecisionRecall(hTestGrado6mala, ytest);
title('Curva Precisión/Recall para regresión grado 6 sin regularizar')

%Mostrar resultados
fprintf('Mejores precisiones para un recall >= 0.9:\n')
fprintf('Regresión básica: %f\n', bestPBasica);
fprintf('Regresión grado 2: %f\n', bestPGrado2);
fprintf('Regresión regularizada grado 6: %f\n', bestPReg);
fprintf('Regresión grado 6 sin regularizar: %f\n', bestPGrado6Mala);
