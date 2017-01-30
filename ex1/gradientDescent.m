function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

n=length(theta);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE 
    t = zeros(length(theta),1);

    for k=1:n

        for i=1:m
            t(k) = t(k) + ((theta'*X(i,:)') - y(i))' .* X(i,k);
        end

        t(k) = (alpha/m)*t(k);

    end
fprintf('%f %f \n', size(theta), size(t));

    theta = theta - t;

fprintf('%f %f \n', size(theta), size(t));

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end


