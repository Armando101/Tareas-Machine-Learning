function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % X*theta  -> Esta es mi funcion de hipotesis. h(x)=theta0+theta1*x
    % Para este caso x tiene dos columnas de las cuales la primera esta
    % llena de 1's (x0), y la segunda son los valores x1 (Primera y unica caracteristia)
    % Esto nos devuelve una matriz de m funciones de hipotesis. Lo que
    % hacemos a continuacion es restar la variable y
    
    % Lo que hacemos a continuacion es hacer una matriz de m X size(x,n+1),
    % donde n es el numero de caracteristicas de nuestro problema. Para
    % este caso n=1 (Poblacion de la ciudad).
    
    % El paso siguiente es multiplicar uno a uno todo por la matriz X
    % Esto porque como se ve en la formula para calcular theta observamos
    % que se multiplica por x_j (carteristica j-esima) que para theta0 por
    % definicion x_0=1 y para theta1 se multiplica por x_1 (Primera y unica caracteristica)
    
    % Por ultimo como se ve en la forumla tenemos una sumatoria, esto nos
    % devuelve un vector de 1 X n+1, por ultimo lo multiplicamos por
    % alpha/m y todo lo que tenemos lo restamos de nuestro anterior valor
    % de theta quedandonos un vector de n X 1 que contiene los valores de
    % theta.
    
    delta = (1/m)*sum(X.*repmat((X*theta - y), 1, size(X,2)));
    
    
    theta = (theta' - (alpha * delta))';

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
end

end
