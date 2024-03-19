% Two approaches; 

% First; trying with a network from scratch
% Classifying features extracted from 30 concatenated frames directly;
% Defining CNN architecture
layers11 = [
    % Input layer
    imageInputLayer([8 8 30]) 
    % Convolutional and pooling layers
    convolution2dLayer([2 2], 16)
    reluLayer()
    maxPooling2dLayer([2 2], 'Stride', [2 2])
    % Fully connected layers
    fullyConnectedLayer(32)
    reluLayer()
    fullyConnectedLayer(12)
    softmaxLayer()
    classificationLayer()
];
% Yields about 33276 parameters, which is less than the imposed limit 936 * 8 * 8

%%
maxEpochs = 2;
miniBatchSize = 5;
options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Plots', 'training-progress', ...
    'Verbose', false);



% Second; Modifying the parameters of an already existing model (alexnet) with good accuracy on imagenet
nn = alexnet;
deepNetworkDesigner(nn)
    
