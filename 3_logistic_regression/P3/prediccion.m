function [y] = prediccion(x, umbral)
%Prediccion binaria de la salida de x
h = sigmoidal(x);
y = (h >= umbral);
end

