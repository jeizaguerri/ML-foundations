function [th,C] = descenso_gradiente_huber(th,X,y,alfa,epsilon,max_iters,d)
%Algoritmo de descenso de gradiente
%alfa:Factor de aprendizaje; epsilon:Umbral de convergencia; d:delta hubber
C = double.empty;
j_old = 0;
for iter = 1:max_iters
  [j,g] =  CosteHuber(th,X,y,d);
  th = th - alfa * g;
  C = [C, j];

  if abs(j - j_old) < epsilon
    iter    %Mostar cuantas iteraciones ha tardado
    break;
  end
  j_old = j;
end
end