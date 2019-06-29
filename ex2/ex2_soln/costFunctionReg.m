function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=size(X,2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================
t=0;
for i=1:m,
h=0;
g=0;
for j=1:n,
      g=g+theta(j)*X(i,j);
end
h=sigmoid(g);
t=((y(i)*log(h))+((1-y(i))*log(1-h))); 
J=J+t;
endfor
J=J*(1/m);
J=J*-1;
l=0;
for i=2:n,
  l=l+theta(i)*theta(i);
endfor
l=(l*lambda)/(2*m);
J=J+l;



for t=1:n,
  ti=0;
for i=1:m,
h=0;
g=0;
for j=1:n,
      g=g+(theta(j)*X(i,j));
end
h=sigmoid(g);
ti=ti+((h-y(i))*X(i,t));
endfor
ti=ti/m;
if(t==1)
    grad(1)=ti;
else
    grad(t)=ti+((lambda/m)*theta(t));
endif

endfor    










