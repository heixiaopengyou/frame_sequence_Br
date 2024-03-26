main_folder_train = '.../Briareo_dataset_mat30/train.zip/';
main_folder_val = '.../Briareo_dataset_mat30/validation.zip/';
main_folder_test = '.../Briareo_dataset_mat30/test.zip/';
ds_train = CustomDatastore(main_folder_train);
ds_val = CustomDatastore(main_folder_val);
ds_test = CustomDatastore(main_folder_test);

%training options
numEpochs = 5;
miniBatchSize = 16;
initialLearnRate = 0.01;
decay = 0.01;
momentum = 0.9;

% Initializing the velocity of the Stochastic Gradient Descent
velocity = [];

% Calculating the total number of iterations for the training progress monitor.
numObservationsTrain = 936; % For training
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

% Initializing the training options
monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info=["Epoch" "LearnRate"], ...
    XLabel="Iteration");

% Defining the network layers
input_size = [8 8 30];
layers = [
    % Input layer
    imageInputLayer(input_size)
    
    % Depthwise separable convolution block 1
    % layer = groupedConvolution2dLayer(filterSize,numFiltersPerGroup,numGroups)
    groupedConvolution2dLayer(3,2,'channel-wise')
    reluLayer
    
    % Depthwise separable convolution block 2
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
];
net = dlnetwork(layers);

