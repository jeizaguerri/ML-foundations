
%% Lab 6.1: PCA 

clear all
close all

% load images 
% images size is 20x20. 

load('MNISTdata2.mat');     %X, y, Xtest, ytest

%Permutar datos
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);

%Constantes
nrows=20;
ncols=20;
nimages = size(X,1);
nclases = 10;

%Show the images
%for I=1:40:nimages, 
%    imshow(reshape(X(I,:),nrows,ncols))
%    pause(0.1)
%end


%% Calculo de los vectores y valores propios

% Obtener valores y vectores propios ordenados
[U_ord, ValoresPropios_ord, mediaX] = pca(X);

% Hacer transformaciones para obtener Zs
Z = transformarPca(X,mediaX,400,U_ord);

% Mostrar Zs en las dos primeras dimensiones (Ya se puede ver algo de separacion)
figure(100)
clf, hold on
plotwithcolor(Z(:,1:2), y);
grid on;
title('Datos etiquetados en función de las 2 primeras dimensiones de Z');
xlabel('Dim1'); ylabel('Dim2');


%% Busqueda de mejor modelo
fprintf('\n*** Busqueda de mejor modelo ***\n');

% Calcular evolucion de la variabilidad acumulada en funcion de k
varTotal = sum(ValoresPropios_ord);
K = (1:length(ValoresPropios_ord));    % Valores con los que se va a probar
Porcentajes = zeros(length(K), 0);
for k = K
    varParcial = sum(ValoresPropios_ord(1:k));
    p = varParcial / varTotal;
    Porcentajes(k) = p;
end
figure;
plot(K, Porcentajes);
grid on;
title('Variabilidad acumulada en funcion de k');
xlabel('k'); ylabel('Lambda acumulado');

% En funcion de los resultados, elegir algunas ks y usar kfold para
% entrenar modelos y comprobar tasas de acierto (y tasa de error proyectiva).
ValoresK = [(1:50), 100, 150, 400];
[tasasAcierto, mejorK, mejorTaVal, erroresReproyeccion ]= kfoldTaK(X, y, ValoresK, 10^-2, 10);

fprintf('Mejor resultado: K = %d;  taVal = %d\n', mejorK, mejorTaVal);

figure;
plot(ValoresK, tasasAcierto(1, :), 'b-');
hold on;
plot(ValoresK, tasasAcierto(2, :), 'r-');
hold off;
legend('Train', 'Validacion');
grid on;
title('Tasa de acierto media en función de k');
xlabel('k'); ylabel('Ta');

figure;
plot(ValoresK, erroresReproyeccion(1, :), 'b-');
hold on;
plot(ValoresK, erroresReproyeccion(2, :), 'r-');
hold off;
legend('Train', 'Validacion');
grid on;
title('Error de reproyección medio en función de k');
xlabel('k'); ylabel('Error');

%% Entrenamiento con la mejor k encontrada
fprintf('\n*** Entrenamiento con k = %d ***\n', mejorK);

lambdaReg = 10^-2;
mejorZtrain = transformarPca(X, mediaX, mejorK, U_ord);
mejorZtest = transformarPca(Xtest, mediaX, mejorK, U_ord);
mejorModelo = entrenarGaussianas(mejorZtrain, y, nclases, 0, lambdaReg);
% Predicciones
yhatTrain = clasificacionBayesiana(mejorModelo, mejorZtrain);
yhatTest = clasificacionBayesiana(mejorModelo, mejorZtest);
% Evaluación
taTrain = tasaAcierto(yhatTrain, y);
taTest = tasaAcierto(yhatTest, ytest);
% Mostrar resultados
fprintf('taTrain: %d;   taVal: %d\n', taTrain, taTest);


%% Mostrar las imágenes utilizando los nuevos atributos para ver qué información conservan
nuevoTamLado = 6;
dimension = nuevoTamLado ^2;
Zreducido = transformarPca(X, mediaX, dimension, U_ord);

figure;
for i =  1:10
    for v = 1:10
        ImagenesValor = X(y == v, :);
        TransformadasValor = Zreducido(y == v, :);
        %subplot(10,10,v), imshow(imresize(reshape(ImagenesValor(i,:),nrows,ncols),2));
        subplot(10,10,(i-1)*10 + v), imshow(imresize(reshape(TransformadasValor(i,:),nuevoTamLado,nuevoTamLado),2));
    end
end
sgtitle(sprintf('Ejemplos de imagenes con reduccion de dimesion D=%d \n(Una clase por columna)',dimension));



%% Probar a reproyectar los datos para comprobar la pérdida de información.

U_ordTrans = U_ord';
Xreproyectadas = mejorZtrain * U_ordTrans(1:mejorK, :) + mediaX;

for clase = 1:nclases
    figure;
    X_clase = X(y==clase,:);
    Xreproy_clase = Xreproyectadas(y == clase, :);
    subplot(1,2,1), imshow(reshape(X_clase(1,:),nrows,ncols))
    title('Original');
    subplot(1,2,2), imshow(reshape(Xreproy_clase(1,:),nrows,ncols))
    title('Reproyectada');
    saveas(gcf, sprintf('%d.png',clase), 'png');
end





