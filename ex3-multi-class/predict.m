function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
% Returning the number label, the confidence, and the second highest confidence with label.
p = zeros(size(X, 1), 4);

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

X = [ones(m, 1) X];
for pred = 1:m
    hiddenLayer = [1 sigmoid(X(pred,:) * Theta1')];
    outputLayer = sigmoid(hiddenLayer * Theta2');
    
    % Get first prediction
    [value, index] = max(outputLayer);
    p(pred, 1:2) = [index value];

    % Get second prediction
    outputLayer(index) = [];
    [value, secondIndex] = max(outputLayer);
    if secondIndex >= index
        % Sum 1 because of the removed index.
        secondIndex += 1;
    end
    p(pred, 3:4) = [secondIndex value];
end



% =========================================================================


end
