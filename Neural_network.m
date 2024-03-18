% Two approaches; 

% First; trying with a network from scratch
% Classifying features extracted from 30 concatenated frames directly;
% Defining CNN architecture
layers = [
    imageInputLayer([8 8 30])
    
    % Convolutional layers (adjusting filters and kernel size for 8 x 8 inputs)
    convolution2dLayer([5,5], 16)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer([1,1], 32)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    flattenLayer
    % Fully connected layers
    fullyConnectedLayer(64)
    fullyConnectedLayer(12, 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10) % Adjusting learning rates for final layer
    softmaxLayer
    classificationLayer];

% Second; Modifying the parameters of an already existing model (alexnet) with good accuracy on imagenet
nn = alexnet;
deepNetworkDesigner(nn)
    
