
%% Lab 6.2: PCA 

clear all
close all

% load images 
% images size is 20x20. 

load('MNISTdata2.mat'); 

%Permutar datos
%rand('state',0);
%p = randperm(length(y));
%X = X(p,:);
%y = y(p);

nrows=20;
ncols=20;

nclases = 10;
nimages = size(X,1);
nimagestest = size(Xtest,1);

% This parameter controls the amount of noise injected per pixel
noise_std =0.3;
X_noise = X + randn(nimages, nrows*ncols)*noise_std;
Xtest_noise = Xtest + randn(nimagestest, nrows*ncols)*noise_std;

Xtest_noise_1 = X + randn(nimages, nrows*ncols)*0.5;
Xtest_noise_2 = X + randn(nimages, nrows*ncols)*0.75;


%Show the noisy images
for I=1:40:nimagestest
    subplot(1,4,1)
    imshow(reshape(Xtest(I,:),nrows,ncols))
    subplot(1,4,2)
    imshow(reshape(Xtest_noise(I,:),nrows,ncols))
    subplot(1,4,3)
    imshow(reshape(Xtest_noise_1(I,:),nrows,ncols))
    subplot(1,4,4)
    imshow(reshape(Xtest_noise_2(I,:),nrows,ncols))
    pause(0.1)
end

% Un ejemplo
figure;
I = 750;
subplot(1,4,1)
imshow(reshape(Xtest(I,:),nrows,ncols))
title(sprintf('%16.2f',0));
subplot(1,4,2)
imshow(reshape(Xtest_noise(I,:),nrows,ncols))
title(sprintf('%16.2f',0.3));
subplot(1,4,3)
imshow(reshape(Xtest_noise_1(I,:),nrows,ncols))
title(sprintf('%16.2f',0.5));
subplot(1,4,4)
imshow(reshape(Xtest_noise_2(I,:),nrows,ncols))
title(sprintf('%16.2f',0.75));



%% Perform PCA (use your code from previous exercise)

% Obtener valores y vectores propios ordenados
[U_ord, ValoresPropios_ord, mediaX] = pca(X);
k = 35;
Z = transformarPca(Xtest_noise, mediaX, k, U_ord);



%% Denoise test images using k components

% Reproyectar
U_ordTrans = U_ord';
Xreproyectadas = Z * U_ordTrans(1:k, :) + mediaX;

% Display original and reconstructed images
figure;
for B=0:40:nimagestest-10
    for I = 1:10
        subplot(2,10,I);
        imshow(reshape(Xtest_noise(B+I,:),nrows,ncols));
        title('Og');
        subplot(2,10,I+10);
        imshow(reshape(Xreproyectadas(B+I,:),nrows,ncols));
        title('Rec');
    end
    pause(0.1)
end

%% Use the classifier from previous labs (P5 and P61) to classify noise 
%  test images both directly and on the PCA space

%Permutar datos
rand('state',0);
p = randperm(length(y));
X = X(p,:);
X_noise = X_noise(p,:);
y = y(p);

% Entrenar con los datos con ruido
fprintf('\n*** Entrenamiento con datos con ruido ***\n');
lambdaReg = 10^-2;
modelo = entrenarGaussianas(X_noise, y, nclases, 0, lambdaReg);
% Predicciones
yhatTrain = clasificacionBayesiana(modelo, X_noise);
yhatTest = clasificacionBayesiana(modelo, Xtest_noise);
% Evaluación
taTrain = tasaAcierto(yhatTrain, y);
taTest = tasaAcierto(yhatTest, ytest);
% Mostrar resultados
fprintf('taTrain: %d;   taVal: %d\n', taTrain, taTest);

% Entrenar aplicando PCA
fprintf('\n*** Entrenamiento con datos reducidos ***\n');
Ztrain = transformarPca(X_noise, mediaX, k, U_ord);
Ztest = transformarPca(Xtest_noise, mediaX, k, U_ord);


figure;
clf, hold on;
plotwithcolor(Ztrain(:,1:2), y);
grid on;
title('Datos etiquetados en función de las 2 primeras dimensiones de Z. Train');
xlabel('Dim1'); ylabel('Dim2');

figure;
clf, hold on;
plotwithcolor(Ztest(:,1:2), ytest);
grid on;
title('Datos etiquetados en función de las 2 primeras dimensiones de Z. Test');
xlabel('Dim1'); ylabel('Dim2');

lambdaReg = 10^-2;
modelo = entrenarGaussianas(Ztrain, y, nclases, 0, lambdaReg);

% Predicciones
yhatTrain = clasificacionBayesiana(modelo, Ztrain);
yhatTest = clasificacionBayesiana(modelo, Ztest);
% Evaluación
taTrain = tasaAcierto(yhatTrain, y);
taTest = tasaAcierto(yhatTest, ytest);
% Mostrar resultados
fprintf('taTrain: %d;   taVal: %d\n', taTrain, taTest);

%% Optional You can try and analyse:
% 1- Propose a strategy to select the # of components (hint: reconstruction
% error for train images)
fprintf('\n*** Búsqueda del mejor k para denoising ***\n');


ValoresK = (1:400);
erroresReproyeccion = zeros(2, length(ValoresK));
mejorK = 0;
mejorErrorTest = inf;
for k = ValoresK
    % Proyectar
    ZkTrain = transformarPca(X_noise, mediaX, k, U_ord);
    ZkTest = transformarPca(Xtest_noise, mediaX, k, U_ord);
    % Error de reproyeccion
    errRepTrain = errorReproyeccion(ZkTrain, X, U_ord, k, mediaX);
    errRepTest = errorReproyeccion(ZkTest, Xtest, U_ord, k, mediaX);
    % Guardar en el vector
    erroresReproyeccion(1, k) = errRepTrain;
    erroresReproyeccion(2, k) = errRepTest;
    % Guardar si es el mejor
    if errRepTest < mejorErrorTest
        mejorErrorTest = errRepTest;
        mejorK = k;
    end
end

% Mostrar resultados
fprintf('\nMejor k = %d;    errorTest=%d\n', mejorK, mejorErrorTest);


figure;
plot(ValoresK, erroresReproyeccion(1,:));
hold on;
plot(ValoresK, erroresReproyeccion(2,:));
hold off;
grid on;
xlabel('k'); ylabel('Error de reproyección');
title('Error de reproyección en función de k');
legend('Train', 'Test');



