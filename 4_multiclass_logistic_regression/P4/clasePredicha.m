function [bestClases] = clasePredicha(x,Thetas)
%Saca la clase más probable utilizando one-vs-all para los datos de entrada
h = sigmoidal(x * Thetas);
[largest, bestClases] =max(h,[],2);

end

