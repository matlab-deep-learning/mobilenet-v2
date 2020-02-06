function lgraph = mobilenetv2Layers()
% mobilenetv2Layers   MobileNet-v2 layer graph
%
% lgraph = mobilenetv2Layers creates a layer graph with the network
% architecture of MobileNet-v2. The layer graph contains no weights.

lgraph = layerGraph();
%% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear 
% array of layers.

tempLayers = [
    imageInputLayer([224 224 3],"Name","input_1","Normalization","zscore")
    convolution2dLayer([3 3],32,"Name","Conv1","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","bn_Conv1","Epsilon",0.001)
    clippedReluLayer(6,"Name","Conv1_relu")
    groupedConvolution2dLayer([3 3],1,32,"Name","expanded_conv_depthwise","Padding","same")
    batchNormalizationLayer("Name","expanded_conv_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","expanded_conv_depthwise_relu")
    convolution2dLayer([1 1],16,"Name","expanded_conv_project","Padding","same")
    batchNormalizationLayer("Name","expanded_conv_project_BN","Epsilon",0.001)
    convolution2dLayer([1 1],96,"Name","block_1_expand","Padding","same")
    batchNormalizationLayer("Name","block_1_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_1_expand_relu")
    groupedConvolution2dLayer([3 3],1,96,"Name","block_1_depthwise","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","block_1_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_1_depthwise_relu")
    convolution2dLayer([1 1],24,"Name","block_1_project","Padding","same")
    batchNormalizationLayer("Name","block_1_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],144,"Name","block_2_expand","Padding","same")
    batchNormalizationLayer("Name","block_2_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_2_expand_relu")
    groupedConvolution2dLayer([3 3],1,144,"Name","block_2_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_2_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_2_depthwise_relu")
    convolution2dLayer([1 1],24,"Name","block_2_project","Padding","same")
    batchNormalizationLayer("Name","block_2_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_2_add")
    convolution2dLayer([1 1],144,"Name","block_3_expand","Padding","same")
    batchNormalizationLayer("Name","block_3_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_3_expand_relu")
    groupedConvolution2dLayer([3 3],1,144,"Name","block_3_depthwise","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","block_3_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_3_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_3_project","Padding","same")
    batchNormalizationLayer("Name","block_3_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","block_4_expand","Padding","same")
    batchNormalizationLayer("Name","block_4_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_4_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_4_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_4_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_4_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_4_project","Padding","same")
    batchNormalizationLayer("Name","block_4_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_4_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","block_5_expand","Padding","same")
    batchNormalizationLayer("Name","block_5_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_5_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_5_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_5_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_5_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_5_project","Padding","same")
    batchNormalizationLayer("Name","block_5_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_5_add")
    convolution2dLayer([1 1],192,"Name","block_6_expand","Padding","same")
    batchNormalizationLayer("Name","block_6_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_6_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_6_depthwise","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","block_6_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_6_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_6_project","Padding","same")
    batchNormalizationLayer("Name","block_6_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_7_expand","Padding","same")
    batchNormalizationLayer("Name","block_7_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_7_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_7_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_7_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_7_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_7_project","Padding","same")
    batchNormalizationLayer("Name","block_7_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_7_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_8_expand","Padding","same")
    batchNormalizationLayer("Name","block_8_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_8_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_8_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_8_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_8_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_8_project","Padding","same")
    batchNormalizationLayer("Name","block_8_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_8_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_9_expand","Padding","same")
    batchNormalizationLayer("Name","block_9_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_9_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_9_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_9_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_9_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_9_project","Padding","same")
    batchNormalizationLayer("Name","block_9_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_9_add")
    convolution2dLayer([1 1],384,"Name","block_10_expand","Padding","same")
    batchNormalizationLayer("Name","block_10_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_10_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_10_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_10_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_10_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_10_project","Padding","same")
    batchNormalizationLayer("Name","block_10_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],576,"Name","block_11_expand","Padding","same")
    batchNormalizationLayer("Name","block_11_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_11_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_11_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_11_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_11_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_11_project","Padding","same")
    batchNormalizationLayer("Name","block_11_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_11_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],576,"Name","block_12_expand","Padding","same")
    batchNormalizationLayer("Name","block_12_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_12_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_12_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_12_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_12_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_12_project","Padding","same")
    batchNormalizationLayer("Name","block_12_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_12_add")
    convolution2dLayer([1 1],576,"Name","block_13_expand","Padding","same")
    batchNormalizationLayer("Name","block_13_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_13_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_13_depthwise","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","block_13_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_13_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_13_project","Padding","same")
    batchNormalizationLayer("Name","block_13_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],960,"Name","block_14_expand","Padding","same")
    batchNormalizationLayer("Name","block_14_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_14_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_14_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_14_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_14_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_14_project","Padding","same")
    batchNormalizationLayer("Name","block_14_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_14_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],960,"Name","block_15_expand","Padding","same")
    batchNormalizationLayer("Name","block_15_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_15_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_15_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_15_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_15_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_15_project","Padding","same")
    batchNormalizationLayer("Name","block_15_project_BN","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_15_add")
    convolution2dLayer([1 1],960,"Name","block_16_expand","Padding","same")
    batchNormalizationLayer("Name","block_16_expand_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_16_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_16_depthwise","Padding","same")
    batchNormalizationLayer("Name","block_16_depthwise_BN","Epsilon",0.001)
    clippedReluLayer(6,"Name","block_16_depthwise_relu")
    convolution2dLayer([1 1],320,"Name","block_16_project","Padding","same")
    batchNormalizationLayer("Name","block_16_project_BN","Epsilon",0.001)
    convolution2dLayer([1 1],1280,"Name","Conv_1")
    batchNormalizationLayer("Name","Conv_1_bn","Epsilon",0.001)
    clippedReluLayer(6,"Name","out_relu")
    globalAveragePooling2dLayer("Name","global_average_pooling2d_1")
    fullyConnectedLayer(1000,"Name","Logits")
    softmaxLayer("Name","Logits_softmax")
    classificationLayer("Name","ClassificationLayer_Logits")];
lgraph = addLayers(lgraph,tempLayers);

%% Connect Layer Branches
% Connect all the branches of the network to create the network graph.

lgraph = connectLayers(lgraph,"block_1_project_BN","block_2_expand");
lgraph = connectLayers(lgraph,"block_1_project_BN","block_2_add/in2");
lgraph = connectLayers(lgraph,"block_2_project_BN","block_2_add/in1");
lgraph = connectLayers(lgraph,"block_3_project_BN","block_4_expand");
lgraph = connectLayers(lgraph,"block_3_project_BN","block_4_add/in2");
lgraph = connectLayers(lgraph,"block_4_project_BN","block_4_add/in1");
lgraph = connectLayers(lgraph,"block_4_add","block_5_expand");
lgraph = connectLayers(lgraph,"block_4_add","block_5_add/in2");
lgraph = connectLayers(lgraph,"block_5_project_BN","block_5_add/in1");
lgraph = connectLayers(lgraph,"block_6_project_BN","block_7_expand");
lgraph = connectLayers(lgraph,"block_6_project_BN","block_7_add/in2");
lgraph = connectLayers(lgraph,"block_7_project_BN","block_7_add/in1");
lgraph = connectLayers(lgraph,"block_7_add","block_8_expand");
lgraph = connectLayers(lgraph,"block_7_add","block_8_add/in2");
lgraph = connectLayers(lgraph,"block_8_project_BN","block_8_add/in1");
lgraph = connectLayers(lgraph,"block_8_add","block_9_expand");
lgraph = connectLayers(lgraph,"block_8_add","block_9_add/in2");
lgraph = connectLayers(lgraph,"block_9_project_BN","block_9_add/in1");
lgraph = connectLayers(lgraph,"block_10_project_BN","block_11_expand");
lgraph = connectLayers(lgraph,"block_10_project_BN","block_11_add/in2");
lgraph = connectLayers(lgraph,"block_11_project_BN","block_11_add/in1");
lgraph = connectLayers(lgraph,"block_11_add","block_12_expand");
lgraph = connectLayers(lgraph,"block_11_add","block_12_add/in2");
lgraph = connectLayers(lgraph,"block_12_project_BN","block_12_add/in1");
lgraph = connectLayers(lgraph,"block_13_project_BN","block_14_expand");
lgraph = connectLayers(lgraph,"block_13_project_BN","block_14_add/in2");
lgraph = connectLayers(lgraph,"block_14_project_BN","block_14_add/in1");
lgraph = connectLayers(lgraph,"block_14_add","block_15_expand");
lgraph = connectLayers(lgraph,"block_14_add","block_15_add/in2");
lgraph = connectLayers(lgraph,"block_15_project_BN","block_15_add/in1");

end