% Two approaches; 

% First; trying with a network from scratch
% Classifying features extracted from 30 concatenated frames directly;
% Defining input size
input_size = [8 8 30];

% Defining the network layers
layers = [
    % Input layer
    imageInputLayer(input_size)
    
    % Depthwise separable convolution block 1
    % layer = groupedConvolution2dLayer(filterSize,numFiltersPerGroup,numGroups)
    % groupedConvolution2dLayer(3, 3, 'DilationRate', [1 1], 'Padding', 'same')
    % pointwiseConv2dLayer(32, 1, 'Padding', 'same')
    % groupedConvolution2dLayer(3,128,3,'Padding','same')
    groupedConvolution2dLayer(3,2,'channel-wise')
    reluLayer
    
    % Depthwise separable convolution block 2
    % groupedConvolution2dLayer(3, 3, 'DilationRate', [1 1], 'Padding', 'same')
    % pointwiseConv2dLayer(64, 1, 'Padding', 'same')
    % groupedConvolution2dLayer(3,128,3,'Padding','same')
    groupedConvolution2dLayer(3,2,'channel-wise')
    reluLayer
    
    % Average pooling layer
    averagePooling2dLayer(2, 'Stride', 2)
    
    % Fully connected layers
    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.5) % Optional dropout for regularization
    fullyConnectedLayer(12) % Output layer with 12 neurons for 12 classes
    softmaxLayer
    classificationLayer()
];

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
    
