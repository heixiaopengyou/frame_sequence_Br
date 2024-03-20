%%
reset(ds_test);
% Training the model
net = trainNetwork(ds_test, layers, options);
