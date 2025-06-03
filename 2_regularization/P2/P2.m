close all;
%% Cargar los datos
datos = load('CochesTrain.txt');
ydatos = datos(:, 1);   % Precio en Euros
Xdatos = datos(:, 2:4); % Años, Km, CV
x1dibu = linspace(min(Xdatos(:,1)), max(Xdatos(:,1)), 100)'; %para dibujar
x2dibu = linspace(min(Xdatos(:,2)), max(Xdatos(:,2)), 100)'; %para dibujar
x3dibu = linspace(min(Xdatos(:,3)), max(Xdatos(:,3)), 100)'; %para dibujar


datos_test = load('CochesTest.txt');
ytest = datos_test(:,1);  % Precio en Euros
Xtest = datos_test(:,2:4); % Años, Km, CV
Ntest = length(ytest);

%% 1. Seleccion de modelos mediante búsqueda heurística
disp('********************1. Seleccion de modelos mediante búsqueda heurística*******************');

%Selección de modelo (Grados del polinomio)
[ Grados, costes ] = kfold_cross_validation_heuristica(5, Xdatos, ydatos, [1 1 1], 10);

%Mostrar modelo elegido
fprintf('Grados: %d, %d, %d\n', Grados(1), Grados(2), Grados(3));

%Mostrar evolución del error RMSE
for i=1:length(costes(:,1,1))
    figure;
    plot(squeeze(costes(i,1,:)),'r-');
    hold on;
    plot(squeeze(costes(i,2,:)),'b-');
    hold off;
    title(sprintf('Evolución del error en atributo %d', i));
    ylabel('Error(€)'); xlabel('Grado atributo');
    legend('Error Entrenamiento', 'Error validacion');
end

%Entrenar modelo final con TODOS los datos (sin partición)
X = expandir(Xdatos, Grados);
[ X_norm, mu, sig ] = normalizar(X);
w_norm = X_norm\ydatos;
w = desnormalizar(w_norm, mu, sig);

%Evaluacion final
X_test = expandir(Xtest, Grados);

RMSEtr = RMSE(w, X, ydatos) %Sobre datos de entrenamiento
MAEtr = MAE(w, X, ydatos)
RMSEtest = RMSE(w, X_test, ytest)   %Sobre datos de test
MAEtest = MAE(w, X_test, ytest)


%% 2. Seleccion de modelos mediante búsqueda exhaustiva (grid search)
disp('********************2. Seleccion de modelos mediante búsqueda exhaustiva (grid search)*******************');

%Selección de modelo (Grados del polinomio)
[ Grados, mejor_error ] = kfold_cross_validation_exhaustiva(5, Xdatos, ydatos, [1 1 1], 10, 1);

fprintf('Mejor polinomio encontrado: Grados (%d, %d, %d)\nCon error RMSE %f', Grados(1), Grados(2), Grados(3), mejor_error);

%Entrenar modelo final con TODOS los datos (sin partición)
X = expandir(Xdatos, Grados);
[ X_norm, mu, sig ] = normalizar(X);
w_norm = X_norm\ydatos;
w = desnormalizar(w_norm, mu, sig);

%Evaluacion final
X_test = expandir(Xtest, Grados);

RMSEtr = RMSE(w, X, ydatos) %Sobre datos de entrenamiento
MAEtr = MAE(w, X, ydatos)
RMSEtest = RMSE(w, X_test, ytest)   %Sobre datos de test
MAEtest = MAE(w, X_test, ytest)

%% 3. Regularización. Progama la búsqueda de un modelo de regresión polinómica de grado
disp('********************3. Regularización. Progama la búsqueda de un modelo de regresión polinómica de grado*******************');

%Utilizando grado 10 para todas las entradas
Grados = [10, 10, 10];
Lambdas = 10.^(-10:3);

%Buscar el mejor valor de lambda
[best_lambda, C] = kfold_cross_validation_regularizacion(5, Xdatos, ydatos, Grados, Lambdas);
best_lambda

%Mostar evolución del coste
figure;
semilogx(Lambdas, C(1,:),'r-');
hold on;
semilogx(Lambdas, C(2,:),'b-');
hold off;
title('Evolución del error con lambda');
ylabel('Error(€)'); xlabel('Lambda');
legend('Error Entrenamiento', 'Error validacion');

%Entrenar el modelo con la lambda elegida
X = expandir(Xdatos, Grados);
[ X_norm, mu, sig ] = normalizar(X);
D = sum(Grados(:));
H = X_norm' * X_norm + best_lambda * diag([0 ones(1,D)]);
w_norm = H \ (X_norm' * ydatos);
w = desnormalizar(w_norm, mu, sig);


%Mostar el efecto suavizador de lambda sobre un solo atributo
for lambda = 10 .^ [-10, -7, 3]

    grado = 10;
    X = expandir(Xdatos, [grado 0 0]);
    [ Xcv, ycv, Xtr, ytr ] = particion( 1, 5, X, ydatos );
    [ Xtr_norm, mu, sig ] = normalizar(Xtr);
    H = Xtr_norm' * Xtr_norm + lambda * diag([0 ones(1,grado)]);
    w_norm = H \ (Xtr_norm' * ytr);
    w = desnormalizar(w_norm,mu,sig);
    
    figure;
    grid on; hold on;
    plot(Xtr(:,2), ytr, 'bx');
    plot(Xcv(:,2), ycv, 'ro');
    title(sprintf('Polinomio grado 10 lambda %d', lambda));
    ylabel('Precio Coches (Euros)'); xlabel('Años');
    Xd = expandir(x1dibu, grado);
    plot(x1dibu, Xd*w, 'r-'); % Dibujo la recta de predicción
    legend('Datos Entrenamiento', 'Datos validacion', 'Prediccion')
end

%Reentrenar con todos los datos
X = expandir(Xdatos, Grados);
[ X_norm, mu, sig ] = normalizar(X);

D = sum(Grados(:));
H = X_norm' * X_norm + best_lambda * diag([0 ones(1,D)]);
w_norm = H \ (X_norm' * ydatos);
w = desnormalizar(w_norm, mu, sig);

%Evaluar modelo
X_test = expandir(Xtest, Grados);

RMSEtr = RMSE(w, X, ydatos) %Sobre datos de entrenamiento
MAEtr = MAE(w, X, ydatos)
RMSEtest = RMSE(w, X_test, ytest)   %Sobre datos de test
MAEtest = MAE(w, X_test, ytest)

%% Modelo con los mejores grados de polinomios y regularización
disp('********************Modelo con los mejores grados de polinomios y regularización*******************');

Grados = [5,6,6];
Lambdas = 10.^(-15:3);

%Buscar el mejor valor de lambda
[best_lambda, C] = kfold_cross_validation_regularizacion(5, Xdatos, ydatos, Grados, Lambdas);
best_lambda

%Reentrenar con todos los datos
X = expandir(Xdatos, Grados);
[ X_norm, mu, sig ] = normalizar(X);

D = sum(Grados(:));
H = X_norm' * X_norm + best_lambda * diag([0 ones(1,D)]);
w_norm = H \ (X_norm' * ydatos);
w = desnormalizar(w_norm, mu, sig);

%Evaluar modelo
X_test = expandir(Xtest, Grados);

RMSEtr = RMSE(w, X, ydatos) %Sobre datos de entrenamiento
MAEtr = MAE(w, X, ydatos)
RMSEtest = RMSE(w, X_test, ytest)   %Sobre datos de test
MAEtest = MAE(w, X_test, ytest)