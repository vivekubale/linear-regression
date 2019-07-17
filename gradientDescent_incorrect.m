function [theta, J_history] = gradientDescent_incorrect(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. However you should update theta_0 fisrt, then
    %               update theto_1.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % Incorrect way
    temp0 = (1/m)*sum(X*theta-y);
    theta(1) = theta(1) - alpha*temp0;
    temp1 = (1/m)*sum((X*theta-y).*X(:,2));
    theta(2) = theta(2) - alpha*temp1;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

% fprintf('Plotting J(theta) vs iterations....');
% figure;
% plot(num_iters,J_history,'-');

end
