function [th,C] = descenso_gradiente(th,X,y,alfa,epsilon,max_iters)
%Algoritmo de descenso de gradiente
%alfa:Factor de aprendizaje; epsilon:Umbral de convergencia
C = double.empty;
j_old = 0;
for iter = 1:max_iters
  [j,g] = CosteL2(th,X,y);
  th = th - alfa * g;
  C = [C, j];

  if abs(j - j_old) < epsilon
    iter    %Mostar cuantas iteraciones ha tardado
    break;
  end
  j_old = j;
end
end