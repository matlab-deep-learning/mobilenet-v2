%% Classify Image Using MobileNet-v2
% This example shows how to classify an image using the MobileNet-v2
% pretrained convolutional neural network.

%   Copyright 2019 The MathWorks, Inc.

% Read an example image.
img = imread("peppers.png");

% The image that you want to classify must have the same size as the input
% size of the network. Resize the image to be 224-by-224 pixels, the input
% size of MobileNet-v2.
img = imresize(img,[224 224]);

% Assemble the pretrained MobileNet-v2 network. Alternatively, you can
% create a pretrained MobileNet-v2 network by installing the Deep Learning
% Toolbox Model for MobileNet-v2 Network support package from the Add-On
% Explorer using the mobilenetv2 function.
net = assembleMobileNetv2;

% Analyze the network architecture.
analyzeNetwork(net)

% Classify the image using the network.
label = classify(net,img);

% Display the image together with the predicted label.
figure
imshow(img)
title(string(label))