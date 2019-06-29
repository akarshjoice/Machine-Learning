function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);
h1=0;
h=0
for i=1:m,
  h1=theta(1)*X(i,1)+theta(2)*X(i,2)+theta(3)*X(i,3);
   h=sigmoid(h1);
   if h >= 0.5,
       p(i)=1;
   else
       p(i)=0;
   endif
endfor

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%







% =========================================================================


end
