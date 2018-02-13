%% How to make a net to classify without training
%% Workaround for creating a net from layers without training
% I want to use a keras network but acn't because the last layer (binary crossentropy) is not supported.
% So instead of importing the network I import the layers, and replace the last one
% with empty classificationLayer. But the only way to collect layers into a network 
% is through training. So we set the learning rate low for the training not
% to modify the net. This workaround is also good when building a net from
% scratch with the aim of using it as is, with no training.
%% Import layers from .h5
layers=importKerasLayers('moshe2.h5','importweights',true);
%% Replace last layer (placeholder)
layers(end)=classificationLayer;
disp(layers)
%% "Train" net 
% We use minimal learning rate and zeros for images, one per category (there are two output classes).  
options = trainingOptions('sgdm','MaxEpochs',1,'InitialLearnRate',realmin('single'),'MiniBatchSize',2);
cats=categorical([0;1]);
net = trainNetwork(zeros(400,400,3,2), cats, layers, options);
%% Are the weights changes by training?
% I test the last layer, you can test the rest
isequal(layers(24).Weights,net.Layers(24).Weights)