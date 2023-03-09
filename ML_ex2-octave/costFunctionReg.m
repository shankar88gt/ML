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
m = length(y); % number of training examples
q = sigmoid(X*theta);
regterm = (lambda/(2*m)) * sum(theta(2:end).^2);
calcErrors = -y.*log(q) - (1 -y).*log(1-q);
J = (1/m)*sum(calcErrors)+regterm;
regterm = [0 transpose((lambda*theta(2:end)/m))];
grad = sum(X.*(q - y)) / m + regterm;
% =============================================================
end
