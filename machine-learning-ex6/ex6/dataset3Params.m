function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

testValue = [ 0.01 0.03 0.1 0.3 1 3 10 30];
testValueCount = length(testValue);
resultats = zeros(64)(:,1:3);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

for Ctest = 1:testValueCount
  for sigmatest = 1:testValueCount
    fprintf('Simulation number %f\n',double((Ctest-1)*testValueCount+sigmatest));
    model= svmTrain(X, y, testValue(Ctest), @(x1, x2) gaussianKernel(x1, x2, testValue(sigmatest)));
    predictions = svmPredict(model, Xval);
    moyenne  = mean ( double ( predictions ~= yval))
    resultats((Ctest-1)*testValueCount+sigmatest,:) = [testValue(Ctest) testValue(sigmatest) moyenne];
  end
end

fprintf ('Best value\n');
[val,idx] = min(resultats(:,3))

resultats(idx,:);
C = resultats(idx, 1);
sigma = resultats(idx, 2);


% =========================================================================

end
