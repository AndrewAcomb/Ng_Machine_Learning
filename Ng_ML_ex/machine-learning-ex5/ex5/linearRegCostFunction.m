function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Note to self: X is a 12x2 matrix, theta is 2x1
h_theta = (X * theta);
reg_theta = theta;
reg_theta(1) = 0;


J = (1/(2*m)) * sum((h_theta - y).^2) + (lambda/(2*m)) * sum(reg_theta .^2);


grad = (1/m) * (X' * (h_theta - y)) + (lambda/m) * reg_theta;
% X is 12x2, h_theta is 2x1
% X' is 2x12, h_theta-y is 12x1



% =========================================================================

grad = grad(:);

end
