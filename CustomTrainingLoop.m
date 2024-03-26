epoch = 0;
iteration = 0;

% Looping over epochs.
while epoch < numEpochs && ~monitor.Stop
    
    epoch = epoch + 1;
    monitor.Progress = 0;

    reset(datata);  % Reset datastore to the beginning for each epoch
    
    % Looping over mini-batches.
    while hasdata(datata) && ~monitor.Stop

        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [data, label] = read(datata);
        % Convert data and labels to dlarray or other appropriate format
        % label
        data = reshape(data, [input_size(1), input_size(2), [], size(data, 3)]);
        data = dlarray(single(data), 'SSCB');
        labels = ['g00';'g01';'g02';'g03';'g04';'g05';'g06';'g07';'g08';'g09';'g10';'g11'];
        label = cellstr(label);
        label = categorical(label);
        % label = grp2idx(label);
        label = onehotencode(label, 1, 'ClassNames', labels);
        [loss,gradients,state] = dlfeval(@modelLoss,net,data,single(label));
        net.State = state;
        
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        
        % Update the network parameters using the SGDM optimizer.
        [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);
        
        % Update the training progress monitor.
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
        monitor.Progress = 100 * iteration/numIterations;
    end
end


function [loss,gradients,state] = modelLoss(net,X,T)

% Forward data through network.
[Y,state] = forward(net,X);

% Calculating cross-entropy loss.
loss = crossentropy(Y,T);

% Calculating gradients of loss with respect to learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end
