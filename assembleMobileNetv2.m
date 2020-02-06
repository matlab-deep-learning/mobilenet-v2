function net = assembleMobileNetv2()
% assembleMobileNetv2   Assemble MobileNet-v2 network
%
% net = assembleMobileNetv2 creates a MobileNet-v2 network with weights
% trained on ImageNet. You can load the same MobileNet-v2 network by
% installing the Deep Learning Toolbox Model for MobileNet-v2 Network
% support package from the Add-On Explorer and then using the mobilenetv2
% function.

%   Copyright 2019 The MathWorks, Inc.

% Download the network parameters. If these have already been downloaded,
% this step will be skipped.

% The files will be downloaded to a file "mobilenetv2Params.mat", in a
% directory "MobileNetv2" located in the system's temporary directory.
dataDir = fullfile(tempdir, "MobileNetv2");
paramFile = fullfile(dataDir, "mobilenetv2Params.mat");
downloadUrl = "http://www.mathworks.com/supportfiles/nnet/data/networks/mobilenetv2Params.mat";

if ~exist(dataDir, "dir")
    mkdir(dataDir);
end

if ~exist(paramFile, "file")
    disp("Downloading pretrained parameters file (13 MB).")
    disp("This may take several minutes...");
    websave(paramFile, downloadUrl);
    disp("Download finished.");
else
    disp("Skipping download, parameter file already exists.");
end

% Load the network parameters from the file mobilenetv2Params.mat.
s = load(paramFile);
params = s.params;

% Create a layer graph with the network architecture of MobileNet-v2.
lgraph = mobilenetv2Layers;

% Create a cell array containing the layer names.
layerNames = {lgraph.Layers(:).Name}';

% Loop over layers and add parameters.
for i = 1:numel(layerNames)
    name = layerNames{i};
    idx = strcmp(layerNames,name);
    layer = lgraph.Layers(idx);
    
    % Assign layer parameters.
    layerParams = params.(name);
    if ~isempty(layerParams)
        paramNames = fields(layerParams);
        for j = 1:numel(paramNames)
            layer.(paramNames{j}) = layerParams.(paramNames{j});
        end
        
        % Add layer into layer graph.
        lgraph = replaceLayer(lgraph,name,layer);
    end
end

% Assemble the network.
net = assembleNetwork(lgraph);

end