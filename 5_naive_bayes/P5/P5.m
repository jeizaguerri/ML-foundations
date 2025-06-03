%% P5 Juan Eizaguerri Serrano

clear ; close all;

%% Carga de datos
% Carga los datos y los permuta aleatoriamente

load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest

%Permutar datos
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);
p = randperm(length(ytest));
Xtest = Xtest(p,:);
ytest = ytest(p);

%Constantes
numClases = 10;

%% Bayes ingenuo
fprintf('\n*** Bayes ingenuo ***\n');

%Elegir parámetro de regularización
Lambdas = 10.^(-10:5);
[bestLamba, TasasAcierto] = elegirLambdaKfold(X, y, numClases, 1, Lambdas, 5);

fprintf('Mejor lambda: %d\n', bestLamba);

figure;
semilogx(Lambdas, TasasAcierto(1, :), 'b-');
hold on;
semilogx(Lambdas, TasasAcierto(2, :), 'r-');
hold off;
grid on;
xlabel('Lambdas'); ylabel('Tasa de acierto');
title('Tasa de acierto en función de lambda Bayes ingenuo ');
legend('Datos Train', 'Datos validacion');

%Entrenamiento con mejor lambda y todos los datos de entrenamiento
modelo = entrenarGaussianas(X, y, numClases, 1, bestLamba);
yhatTr = clasificacionBayesiana(modelo, X);
yhatTest = clasificacionBayesiana(modelo, Xtest);

%Tasa de acierto
taTr = tasaAcierto(yhatTr, y);
taTest = tasaAcierto(yhatTest, ytest);
fprintf('TaTr= %f     TaTest=%f\n', taTr, taTest');

%Matriz de confusión
cTr = confusionmat(y, yhatTr);
figure;
confusionchart(cTr);
title('Matriz de confusion Bayes ingenuo (Train)');

cTest = confusionmat(ytest, yhatTest);
figure;
confusionchart(cTest);
title('Matriz de confusion Bayes ingenuo (Test)');

%% Covarianzas completas 
fprintf('\n*** Covarianzas completas ***\n');

%Elegir parámetro de regularización
Lambdas = 10.^(-10:5);
[bestLamba, TasasAcierto] = elegirLambdaKfold(X, y, numClases, 0, Lambdas, 5);

fprintf('Mejor lambda: %d\n', bestLamba);

figure;
semilogx(Lambdas, TasasAcierto(1, :), 'b-');
hold on;
semilogx(Lambdas, TasasAcierto(2, :), 'r-');
hold off;
grid on;
xlabel('Lambdas'); ylabel('Tasa de acierto');
title('Tasa de acierto en función de lambda covarianzas completas');
legend('Datos Train', 'Datos validacion');

%Entrenamiento con mejor lambda y todos los datos de entrenamiento
modelo = entrenarGaussianas(X, y, numClases, 0, bestLamba);
yhatTr = clasificacionBayesiana(modelo, X);
yhatTest = clasificacionBayesiana(modelo, Xtest);

%Tasa de acierto
taTr = tasaAcierto(yhatTr, y);
taTest = tasaAcierto(yhatTest, ytest);
fprintf('TaTr= %f     TaTest=%f\n', taTr, taTest');

%Matriz de confusión
cTr = confusionmat(y, yhatTr);
figure;
confusionchart(cTr);
title('Matriz de confusion covarianzas completas (Train)');

cTest = confusionmat(ytest, yhatTest);
figure;
confusionchart(cTest);
title('Matriz de confusion covarianzas completas (Test)');

%Ejemplos de confusiones
verConfusiones(X, y, yhatTr);
title('Ejemplos errores covarianzas completas (Train)');
verConfusiones(Xtest, ytest, yhatTest);
title('Ejemplos errores covarianzas completas (Test)');

%Precisión recall para ver números problemáticos
for claseSel = (1:numClases)
    ytestClase = (ytest == claseSel);
    yPredTestClase = (yhatTest == claseSel);

    pTest = precision(yPredTestClase, ytestClase);
    rTest = recall(yPredTestClase, ytestClase);

    fprintf('Clase:%d:  pTest=%f    rTest=%f\n', claseSel, pTest, rTest);
    fprintf('Clasificados como %d: %d\n', claseSel, sum(yPredTestClase));
end

%Diagrama de barras de tasa de fallos
Barras = zeros(numClases,2);
for claseSel = (1:numClases)
    ytestClase = (ytest == claseSel);
    yPredTestClase = (yhatTest == claseSel);
    nClase = sum(ytestClase);

    fp = sum(yPredTestClase &  (~ytestClase));
    fn = sum((~yPredTestClase) & ytestClase);
    Barras(claseSel, 1) = fp / nClase;
    Barras(claseSel, 2) = fn / nClase;
end
figure;
bar(Barras);
grid on;
title('Porcentaje de errores para datos de test covarianzas completas');
legend('FP', 'FN');


%% Medias y varianzas de los píxeles:
for c = (1:numClases)
    Xc = X(y==c, :);
    %Media
    figure;
    Medias = mean(Xc);
    MediasMostrar = reshape(Medias, 20, 20);
    imagesc(MediasMostrar);
    title(sprintf('Medias clase %d', c));
    colorbar();
    colormap(jet(256))
    %Varianza
    figure;
    Varianzas = var(Xc);
    VarianzasMostrar = reshape(Varianzas, 20, 20);
    imagesc(VarianzasMostrar);
    title(sprintf('Varianzas clase %d', c));
    colorbar();
    colormap(jet(256))
end

%% Matriz de correlaciones para ver independencia de los atributos
Correlaciones = corrcoef(X);
figure;
imagesc(abs(Correlaciones));
colorbar();
colormap(jet(256))
title('Matriz de correlaciones datos de entrenamiento');