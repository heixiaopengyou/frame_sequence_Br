%%
reset(ds_train);
% Training the model
net = trainNetwork(ds_train, layers11, options);
