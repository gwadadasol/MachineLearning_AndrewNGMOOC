function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

uns = ones (m,1);


for i = 1:K
  multiplicateur = (idx ==  uns*i); % create a vector that contains 1 for each line equals to a given centroid
  
  % use the vector to vompute the average value of the values of all the data points feature by features
  % multiplicateur'*X =  sum of the values of the points assign to the given centroid, feature by feature
  % (1 / (sum( multiplicateur) = value to compute the average value
  somme = sum( multiplicateur); 
  
  centroids(i,:) = (multiplicateur'*X);
  
  if (somme != 0)
    centroids(i,:) = (multiplicateur'*X).* (1 / (sum( multiplicateur)));
   endif
end




% =============================================================


end

