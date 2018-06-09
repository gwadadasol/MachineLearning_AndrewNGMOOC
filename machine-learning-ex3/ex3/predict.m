function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Compute the output value of the Hidden Layer
% A = X * Theta' 
% Dimension: (5000x25)  = ( 5000x401) * (401x25)
v = sigmoid (X*Theta1');

% Add the first column for the bias
v = [ones(size(v,1), 1) v];

% Compute the output value of the output Layer
% H = A * Theta2' 
% Dimension: (5000x10) = (5000x26) * (26x10)
w = sigmoid (v*Theta2');


% return  which number has been identified for the given image (1 row from X)
% need to check the position of the highest value in the vector of one row of w 
% the vector c contains the position of the highet value for each row 
[r, c] = max (w,[],2);

% the position is a vlaue between 1 and 10. 
% if the position is 1 then the value identified is 1 
% if the position is 2 then the value identified is 2 
% ...
% if the position is 9 then the value identified is 9 
% if the position is 10 then the value identified is 0 

p = mod ( c,10); 










% =========================================================================


end
