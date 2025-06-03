function [errMedio] = errorReproyeccion(Z, X, U_ord, k, mediaX)
% Calcula el error medio de reproyección de los datos Z al aplicarles la
% transformación U frente a los datos originales X.

UTrans = U_ord';
Xrec = Z * UTrans(1:k, :) + mediaX;
errMedio = norm(X - Xrec, 'fro')^2;

end

