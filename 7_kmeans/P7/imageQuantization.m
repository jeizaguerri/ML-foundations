clear;
close all;

figure;
im = imread('smallparrot.jpg');
imshow(im)

%% datos
D = double(reshape(im,size(im,1)*size(im,2),3));

% Visualizar dimensiones
figure;
scatter3(D(:,1), D(:,2), D(:,3), 5, D./255, 'filled');
title('Pixeles de la imagen original');
%% dimensiones
m = size(D,1);
n = size(D,2);

%% Loro
fprintf('\n*** Imagen loro ***\n');

K = 16;
% Inicializar datos aleatorios
Dunicos = unique(D, 'rows');
r = randi(size(Dunicos, 1), 1, K);
mu0 = Dunicos(r,:);

%bucle kmeans
[mu, c] = kmeans(D, mu0);

% Calcular error
d = distorsion(D,mu,c);
fprintf('Distorsión con con K=%d: %f\n',K, d);

% reconstruir imagen
qIM=zeros(length(c),3);
for h=1:K, ind=find(c==h);
    qIM(ind,:)=repmat(mu(h,:),length(ind),1);
end
qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
figure;
imshow(uint8(qIM));


% Visualizar dimensiones
qImPlano = reshape(qIM, [], 3);
figure;
scatter3(D(:,1), D(:,2), D(:,3), 5, qImPlano./255, 'filled');
title('Pixeles de la imagen comprimida');

%% Luna
fprintf('\n*** Imagen luna ***\n');

figure;
im = imread('luna.jpg');
imshow(im)
D = double(reshape(im,size(im,1)*size(im,2),3));
% Visualizar dimensiones
figure;
scatter3(D(:,1), D(:,2), D(:,3), 5, D./255, 'filled');
title('Pixeles de la imagen original');

K = 16;
% Inicializar datos aleatorios
Dunicos = unique(D, 'rows');
r = randi(size(Dunicos, 1), 1, K);
mu0 = Dunicos(r,:);

%bucle kmeans
[mu, c] = kmeans(D, mu0);

% Calcular error
d = distorsion(D,mu,c);
fprintf('Distorsión con con K=%d: %f\n',K, d);

% reconstruir imagen
qIM=zeros(length(c),3);
for h=1:K, ind=find(c==h);
    qIM(ind,:)=repmat(mu(h,:),length(ind),1);
end
qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
figure;
imshow(uint8(qIM));


% Visualizar dimensiones
qImPlano = reshape(qIM, [], 3);
figure;
scatter3(D(:,1), D(:,2), D(:,3), 5, qImPlano./255, 'filled');
title('Pixeles de la imagen comprimida');

%% Colores
fprintf('\n*** Imagen colores ***\n');

figure;
im = imread('paisaje.jpg');
imshow(im)
D = double(reshape(im,size(im,1)*size(im,2),3));
% Visualizar dimensiones
figure;
scatter3(D(:,1), D(:,2), D(:,3), 5, D./255, 'filled');
title('Pixeles de la imagen original');

K = 16;
% Inicializar datos aleatorios
Dunicos = unique(D, 'rows');
r = randi(size(Dunicos, 1), 1, K);
mu0 = Dunicos(r,:);

%bucle kmeans
[mu, c] = kmeans(D, mu0);

% Calcular error
d = distorsion(D,mu,c);
fprintf('Distorsión con con K=%d: %f\n',K, d);

% reconstruir imagen
qIM=zeros(length(c),3);
for h=1:K, ind=find(c==h);
    qIM(ind,:)=repmat(mu(h,:),length(ind),1);
end
qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
figure;
imshow(uint8(qIM));


% Visualizar dimensiones
qImPlano = reshape(qIM, [], 3);
figure;
scatter3(D(:,1), D(:,2), D(:,3), 5, qImPlano./255, 'filled');
title('Pixeles de la imagen comprimida');

%% Determinar mejor valor de k loro

% datos
im = imread('smallparrot.jpg');
D = double(reshape(im,size(im,1)*size(im,2),3));

valoresK = (2:40);
distorsiones = zeros(1,39);
for k = valoresK
    % Inicializar o datos aleatorios
    Dunicos = unique(D, 'rows');
    r = randi(size(Dunicos, 1), 1, k);
    mu0 = Dunicos(r,:);
    
    %bucle kmeans
    [mu, c] = kmeans(D, mu0);

    %Evaluar
    d = distorsion(D,mu, c);
    distorsiones(k-1) = d;
end
figure;
plot(valoresK, distorsiones);
title('Función de distorsión en función del número de clusters k');
xlabel('K(Número de clusters)'); ylabel('J(Función de distorsión)');
grid on;

%% Determinar mejor valor de k luna

% datos
im = imread('luna.jpg');
D = double(reshape(im,size(im,1)*size(im,2),3));

valoresK = (2:40);
distorsiones = zeros(1,39);
for k = valoresK
    % Inicializar o datos aleatorios
    Dunicos = unique(D, 'rows');
    r = randi(size(Dunicos, 1), 1, k);
    mu0 = Dunicos(r,:);
    
    %bucle kmeans
    [mu, c] = kmeans(D, mu0);

    %Evaluar
    d = distorsion(D,mu, c);
    distorsiones(k-1) = d;
end
figure;
plot(valoresK, distorsiones);
title('Función de distorsión en función del número de clusters k');
xlabel('K(Número de clusters)'); ylabel('J(Función de distorsión)');
grid on;

%% Determinar mejor valor de k paisaje

% datos
im = imread('paisaje.jpg');
D = double(reshape(im,size(im,1)*size(im,2),3));

valoresK = (2:40);
distorsiones = zeros(1,39);
for k = valoresK
    % Inicializar o datos aleatorios
    Dunicos = unique(D, 'rows');
    r = randi(size(Dunicos, 1), 1, k);
    mu0 = Dunicos(r,:);
    
    %bucle kmeans
    [mu, c] = kmeans(D, mu0);

    %Evaluar
    d = distorsion(D,mu, c);
    distorsiones(k-1) = d;
end
figure;
plot(valoresK, distorsiones);
title('Función de distorsión en función del número de clusters k');
xlabel('K(Número de clusters)'); ylabel('J(Función de distorsión)');
grid on;