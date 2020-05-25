function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%


J = (1/m) * sum(-y .* log(sigmoid(X*theta)) - (1 - y) .* log(1 - sigmoid(X*theta)));
% I'm calculating the cost of the hypothesis with my current thetas for logistic regression.
% If X * theta (t0 + t1 * x1 + t2 * x2) is positive, sigmoid of it will be closer to 1 and so the prediction will be 1.
% Else, sigmoid of it will be closer to 0 and so the prediction will be 0.
% Our thetas therefore should be driven in a way that X * theta are positive or negative values. Our cost function and gradient descent will assure so.

% Clarifying log application: when value is close to zero, log will be close to negative infinite. When value is close to 1, log will be close to 0.
% Therefore, the application of log assures the cost will be high when the prediction is wrong in this formula.

grad = (1/m) * sum((sigmoid(X*theta) - y) .* X);
% Cost function derived.

% =============================================================

end
