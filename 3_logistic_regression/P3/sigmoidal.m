function [h] = sigmoidal(z)
%Funcion sigmoidal
e = exp(1);
h = 1 ./ (1 + e .^ (-z));
end

