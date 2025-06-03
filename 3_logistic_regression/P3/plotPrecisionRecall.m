function [mejorRecall90] = plotPrecisionRecall(hPred,yReal)
%Muestra la grafica precision/recall. Devuelve el mejor recall con la
%que hay una precisión mayor o igual que 0.9
V = zeros(100,2);
i = 0;
mejorRecall90 = 0;
for umbral = 1 : -0.01 : 0
    i = i+1;
    yPred = prediccion(hPred, umbral);
    r = recall(yPred,yReal);
    p = precision(yPred, yReal);
    V(i, 1) = r;
    V(i, 2) = p;

    if(p > 0.9 && r > mejorRecall90)
        mejorRecall90 = r;
    end
end
figure;
plot(V(:,1), V(:,2));
grid on;
xlabel('Recall'); ylabel('Precision');
end

