close all;
%% Cargar los datos de entrenamiento y de test
datos = load('PisosTrain.txt');
y = datos(:,3);  % Precio en Euros
x1 = datos(:,1); % m^2
x2 = datos(:,2); % Habitaciones
N = length(y);

datostest = load('PisosTest.txt');
ytest = datostest(:,3);  % Precio en Euros
x1test = datostest(:,1); % m^2
x2test = datostest(:,2); % Habitaciones
Ntest = length(ytest);

%% Apartado 2. Ajuste monovariable con ecuación normal
figure;
plot(x1, y, 'bx');
title('Precio de los Pisos')
ylabel('Euros'); xlabel('Superficie (m^2)');
grid on; hold on; 

X = [ones(N,1) x1];
th = X \ y;  % Pongo un valor cualquiera de pesos
Xextr = [1 min(x1)  % Predicción para los valores extremos
         1 max(x1)];
yextr = Xextr * th;  
plot(Xextr(:,2), yextr, 'r-'); % Dibujo la recta de predicción
legend('Datos Entrenamiento', 'Prediccion')

%Evaluación
Xtest = [ones(Ntest,1) x1test];
error_mono_normal = mre(Xtest,ytest,th,Ntest)

%% Apartado 3. Ajuste multivariable con ecuación normal
X = [ones(N,1) x1 x2];
th = X\y;  % Pongo un valor cualquiera de pesos
yest = valor_esperado(X,th);

% Dibujar los puntos de entrenamiento y su valor estimado 
figure;  
plot3(x1, x2, y, '.r', 'markersize', 20);
axis vis3d; hold on;
plot3([x1 x1]' , [x2 x2]' , [y yest]', '-b');

% Generar una retícula de np x np puntos para dibujar la superficie
np = 20;
ejex1 = linspace(min(x1), max(x1), np)';
ejex2 = linspace(min(x2), max(x2), np)';
[x1g,x2g] = meshgrid(ejex1, ejex2);
x1g = x1g(:); %Los pasa a vectores verticales
x2g = x2g(:);

% Calcula la salida estimada para cada punto de la retícula
Xg = [ones(size(x1g)), x1g, x2g];
yg = valor_esperado(Xg, th);

% Dibujar la superficie estimada
surf(ejex1, ejex2, reshape(yg,np,np)); grid on; 
title('Precio de los Pisos')
zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');

%Evaluación
Xtest = [ones(Ntest,1) x1test x2test];
error_multi_normal = mre(Xtest,ytest,th,Ntest)

%% Apartado 4. Regresión monovariable con descenso de gradiente

%Entradas
X = [ones(N,1) x1];

th = rand(2,1); % Valor inicial aleatorio

% Entrenamiento
[th, C] = descenso_gradiente(th,X,y,0.0000001,0.01,10000000);

% Mostrar evolución del coste
figure;
semilogx(C);
ylabel('coste'); xlabel('iteracion')
title('Evolución del coste durante el entrenamiento')
grid on; hold on;

% Mostrar modelo
figure;
plot(x1, y, 'bx');
title('Precio de los Pisos')
ylabel('Euros'); xlabel('Superficie (m^2)');
grid on; hold on; 

Xextr = [1 min(x1)  % Predicción para los valores extremos
         1 max(x1)];
yextr = Xextr * th;  
plot(Xextr(:,2), yextr, 'r-'); % Dibujo la recta de predicción
legend('Datos Entrenamiento', 'Prediccion')

%Evaluación
Xtest = [ones(Ntest,1) x1test];
error_mono_descenso = mre(Xtest,ytest,th,Ntest)

%% Apartado 5. Ajuste multivariable con descenso de gradiente

X = [ones(N,1) x1 x2];
th = rand(3,1); % Valor inicial aleatorio

% Entrenamiento
[th, C] = descenso_gradiente(th,X,y,0.0000001,0.01,10000000);

% Mostrar evolución del coste
figure;
semilogx(C);
ylabel('coste'); xlabel('iteracion')
title('Evolución del coste durante el entrenamiento')
grid on; hold on;

% Mostrar modelo
yest = valor_esperado(X,th);

% Dibujar los puntos de entrenamiento y su valor estimado 
figure;  
plot3(x1, x2, y, '.r', 'markersize', 20);
axis vis3d; hold on;
plot3([x1 x1]' , [x2 x2]' , [y yest]', '-b');

% Generar una retícula de np x np puntos para dibujar la superficie
np = 20;
ejex1 = linspace(min(x1), max(x1), np)';
ejex2 = linspace(min(x2), max(x2), np)';
[x1g,x2g] = meshgrid(ejex1, ejex2);
x1g = x1g(:); %Los pasa a vectores verticales
x2g = x2g(:);

% Calcula la salida estimada para cada punto de la retícula
Xg = [ones(size(x1g)), x1g, x2g];
yg = valor_esperado(Xg, th);

% Dibujar la superficie estimada
surf(ejex1, ejex2, reshape(yg,np,np)); grid on; 
title('Precio de los Pisos')
zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');

%Evaluación
Xtest = [ones(Ntest,1) x1test x2test];
error_multi_descenso = mre(Xtest,ytest,th,Ntest)

%% Apartado 6. Ajuste multivariable con descenso de gradiente y regresión robusta

X = [ones(N,1) x1 x2];
th = rand(3,1); % Valor inicial aleatorio

% Entrenamiento
[th, C] = descenso_gradiente_huber(th,X,y,0.0000001,0.01,100000000,0.5);

% Mostrar evolución del coste
figure;
semilogx(C);
ylabel('coste'); xlabel('iteracion')
title('Evolución del coste durante el entrenamiento')
grid on; hold on;

% Mostrar modelo
yest = valor_esperado(X,th);

% Dibujar los puntos de entrenamiento y su valor estimado 
figure;  
plot3(x1, x2, y, '.r', 'markersize', 20);
axis vis3d; hold on;
plot3([x1 x1]' , [x2 x2]' , [y yest]', '-b');

% Generar una retícula de np x np puntos para dibujar la superficie
np = 20;
ejex1 = linspace(min(x1), max(x1), np)';
ejex2 = linspace(min(x2), max(x2), np)';
[x1g,x2g] = meshgrid(ejex1, ejex2);
x1g = x1g(:); %Los pasa a vectores verticales
x2g = x2g(:);

% Calcula la salida estimada para cada punto de la retícula
Xg = [ones(size(x1g)), x1g, x2g];
yg = valor_esperado(Xg, th);

% Dibujar la superficie estimada
surf(ejex1, ejex2, reshape(yg,np,np)); grid on; 
title('Precio de los Pisos')
zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');

%Evaluación
Xtest = [ones(Ntest,1) x1test x2test];
error_multi_descenso_robusta = mre(Xtest,ytest,th,Ntest)
