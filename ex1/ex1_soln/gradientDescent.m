function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
e=0;
e1=0;
p=0;
q=0;
t=0;
alpha=alpha/m;
for iter = 1:num_iters
          for i=1:m,
           p =(theta(1)*X(i,1) + theta(2)*X(i,2));
           e = e + ((p - y(i))*1);
           e1= e1 + ((p - y(i))*X(i,2));
          endfor
          
          q=theta(1)-alpha*e;
          t=theta(2)-alpha*e1;
          theta(1)=q;
          theta(2)=t;
          e=0; 
          e1=0;
          p=0;
          q=0;
          t=0; 
         
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
end
