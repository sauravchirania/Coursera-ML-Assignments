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

arr=X*theta;
arr=sigmoid(arr);
grad=arr;
for i=1:m
    if (y(i)==1)
        J=J-log(arr(i));
    else
        J=J-log(1-arr(i));
    end
end

rterm=(sum(theta.^2)-(theta(1)^2));
J=J+((lambda*rterm)/2);
J=J/m;
grad=grad-y;
grad=X'*grad;
theta2=theta;
theta2(1)=0;
grad=grad+lambda*theta2;
grad=grad/m;

% =============================================================

end
