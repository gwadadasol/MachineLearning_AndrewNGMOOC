function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


% compute the mean by column  
mu = (mean(X))';

% 1. Create a matrix containing the vector mu on each line
% 2. substract mu Matrix to X, element by element
% 3. apply power 2 function to each element
% 4. Compute the mean of (x(i) - mu)^2, for each column of X
sigma2 = (mean((X .- repmat(mu', m,1)).^2))';


% =============================================================





end
