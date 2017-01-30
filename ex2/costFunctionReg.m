function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

t1 = 0;
for f=2:size(theta)(1)
t1 = t1 + theta(f)*theta(f);
end
t1 = t1 * (lambda/(2*size(y)(1)));

for i = 1:size(y)
J = J + (-y(i) * log(sigmoid(theta' * X(i,:)')) - (1-y(i)) * log(1-sigmoid(theta' * X(i,:)')));
end
J = J / size(y)(1);
J = J + t1;

t2 = zeros(size(theta));
for g = 2:size(theta)(1)
t2(g) = lambda * theta(g) / size(y)(1);
end

for j = 2:size(theta)
for k = 1:size(y)
grad(j) = grad(j) + ( (sigmoid(theta' * X(k,:)') - y(k)) * X(k,j) );
end
grad(j) = grad(j) / size(y)(1);
grad(j) = grad(j) + t2(j);
end

grad(1) =0;
for k = 1:size(y)
grad(1) = grad(1) + ( (sigmoid(theta' * X(k,:)') - y(k)) * X(k,1) );
end
grad(1) = grad(1) / size(y)(1);


% =============================================================

end
