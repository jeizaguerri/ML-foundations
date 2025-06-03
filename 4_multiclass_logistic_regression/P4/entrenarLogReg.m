function [Thetas] = entrenarLogReg(X, y, lambda, nClases)
%Entrena usanado coste logística regularizado con datos de entrada X,
%salida y y lambda
options.useMex = 1 ;
options.display = 'none';
options.method = 'lbfgs';

yClases = y == (1:nClases);
Thetas = zeros(size(X,2), size(yClases, 2));

Clases = (1:size(yClases, 2));
for clase = Clases
    y = yClases(:, clase);
    th_ini = zeros(size(X,2), 1);
    th = minFunc(@CosteLogReg, th_ini, options, X, y, lambda);
    Thetas(:,clase) = th;
end
end