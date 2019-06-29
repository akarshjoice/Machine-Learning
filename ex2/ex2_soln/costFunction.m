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
J=0;
h=0;
t=0;
h1=0;
for i=1:m,
   h1=theta(1)*X(i,1)+theta(2)*X(i,2)+theta(3)*X(i,3);
   h=sigmoid(h1);
   t=((y(i)*log(h))+((1-y(i))*log(1-h))); 
   J=J+t;
endfor
J=J*(1/m);
J=J*-1;

t1=0;
t2=0;
t3=0;
for i=1:m
   h1=theta(1)*X(i,1)+theta(2)*X(i,2)+theta(3)*X(i,3);
   h=sigmoid(h1);
   t1=t1+(h-y(i))*X(i,1);
   t2=t2+(h-y(i))*X(i,2);
   t3=t3+(h-y(i))*X(i,3);
endfor
grad(1)=t1*(1/m);
grad(2)=t2*(1/m);
grad(3)=t3*(1/m);








% =============================================================

end
