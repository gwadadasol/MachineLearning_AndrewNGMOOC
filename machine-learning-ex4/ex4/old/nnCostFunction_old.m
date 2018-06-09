function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% Non regularized function
% ------------------------
%J = (1 / m) *( (-y' * log(sigmoid(X * theta)))- (1-y')*log(1 - sigmoid(X * theta)));

%grad = (1 / m)*X'*(sigmoid(X*theta)-y);


% 1. Add the 1 column to X ( = A1) 
X = [ones(m, 1) X];

% 2. Compute A2
A2 = sigmoid(X * Theta1');

% 3. Add the 1 column to A2
A2 = [ones(size(A2,1),1) A2];

%4 Compute A3
A3 = sigmoid(A2 * Theta2');

%fprintf('A2=\n');
%size (A2,1)
%size (A2,2)
%fprintf('A3=\n');
%size (A3,1)
%size (A3,2)
%fprintf('y=\n');
%size (y,1)
%size (y,2)

% 5. Compute the sum for each value of K
% 5a. Create a matrix yy where each value of y is replace by a vector of Zero and one 1. 
%     The position of the 1 in the vector is the value of y(i)
yy = zeros(m,10);

for i=1:m,
	yy(i,y(i,1)) = 1;
end;


%fprintf('yy=\n');
%size (yy,1)
%size (yy,2)

% 5b. Compute the matrix xx and sum the value in the diagonale ( we dont need the other values)
%     - the first raw of log (A3) will multiply by the first column of the transpose(yy) ( this is the value of y(1) express
%     with the class K = 10 -> [ 1 0 0 0 0 0 0 0 0 0]. the result is at theposition xx (1,1) 
%     - the second raw of log (A3) will multiply by the second column of the transpose(yy) ( this is the value of y(1) express
%     with the class K = 1 -> [ 0 1 0 0 0 0 0 0 0 0]. the result is at the position xx(2,2)
%     - the third raw of log (A3) will multiply by the third column of the transpose(yy) ( this is the value of y(1) express
%     with the class K = 2 -> [ 0 0 1 0 0 0 0 0 0 0]. the result is at the position xx(2,2)
%   ...and so on
xx = (1 / m) *( (-yy' * log(A3))- (1-yy')*log(1 - A3));
diag(xx);

% 5c. Sum the value of the Diagonal to get J
J = sum (diag(xx));

%size (J, 1)
%size (J, 2)


% Regularized function
% ------------------------

%fprintf('Theta1=\n');
%size (Theta1,1)
%size (Theta1,2)

%fprintf('Theta2=\n');
%size (Theta2,1)
%size (Theta2,2)

%fprintf('lambda =\n');
%size (lambda,1)
%size (lambda,2)

Theta1NoBias = Theta1(:,2:end);
Theta2NoBias = Theta2(:,2:end);


Theta1NoBiasSquare = Theta1NoBias.^2;
Theta2NoBiasSquare = Theta2NoBias.^2;

RegularizedValue = lambda/(2*m) * (sum(sum(Theta1NoBiasSquare)) + sum(sum(Theta2NoBiasSquare)));

J = J + RegularizedValue;

%grad = (1 / m)*X'*(sigmoid(X*theta)-y)+lambda/m*(thetanonregul);

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
