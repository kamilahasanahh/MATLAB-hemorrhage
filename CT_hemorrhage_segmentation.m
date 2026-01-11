clc;
clear;
close all;

% Define the dataset folder
datasetFolder = 'Patients_CT';
imds = imageDatastore(datasetFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Define labels based on folder names
labels = cellfun(@(x) contains(x, 'no'), imds.Files);
labels = categorical(labels, [true false], {'No_Hemorrhage', 'Hemorrhage'});
imds.Labels = labels;

% Helper function to preprocess images
function [Iout, features] = preprocessImage(I)
    % Convert to grayscale if the image is RGB
    if size(I, 3) == 3
        I = rgb2gray(I);
    end

    % Display original image
    subplot(2, 3, 1);
    imshow(I);
    title('Original Image');

    % Apply Wiener filtering to remove noise
    Iwiener = wiener2(I, [3 3]);
    subplot(2, 3, 2);
    imshow(Iwiener);
    title('Wiener Filtered');
    
    % Apply anisotropic diffusion filtering for smoothing
    Ismoothed = imdiffusefilt(Iwiener);
    subplot(2, 3, 3);
    imshow(Ismoothed);
    title('Anisotropic Diffusion Filtered');

    % Apply CLAHE to enhance the contrast
    Iclahe = adapthisteq(Ismoothed);
    subplot(2, 3, 4);
    imshow(Iclahe);
    title('CLAHE Enhanced');

    % Extract HOG features
    features = extractHOGFeatures(Iclahe);
    
    % Output preprocessed image
    Iout = Iclahe;
    subplot(2, 3, 5);
    imshow(Iout);
    title('Preprocessed Image');
end

% Preprocess the images and extract features
numImages = numel(imds.Files);
features = [];
labels = [];

for i = 1:numImages
    img = readimage(imds, i);
    [~, featureVector] = preprocessImage(img);
    features = [features; featureVector];
    labels = [labels; imds.Labels(i)];
end

% Convert labels to numeric values (assuming 'No_Hemorrhage' is 1 and 'Hemorrhage' is 0)
numericLabels = double(labels) - 1; % Convert to 0 and 1

% Split the dataset
% Get indices for each class
noIdx = find(numericLabels == 1);
yesIdx = find(numericLabels == 0);

% Define the proportions for splitting
train_percent = 0.6;
val_percent = 0.2;
test_percent = 0.2;

% Calculate the number of samples for each set
num_samples = numel(imds.Files);
num_train = round(train_percent * num_samples);
num_val = round(val_percent * num_samples);
num_test = num_samples - num_train - num_val;

% Shuffle indices for each class
rng(1); % For reproducibility
noIdx_shuffled = noIdx(randperm(numel(noIdx)));
yesIdx_shuffled = yesIdx(randperm(numel(yesIdx)));

% Assign indices for each set
trainIdx = [noIdx_shuffled(1:round(0.6*numel(noIdx_shuffled))); ...
            yesIdx_shuffled(1:round(0.6*numel(yesIdx_shuffled)))];
valIdx = [noIdx_shuffled(round(0.6*numel(noIdx_shuffled))+1:round(0.8*numel(noIdx_shuffled))); ...
          yesIdx_shuffled(round(0.6*numel(yesIdx_shuffled))+1:round(0.8*numel(yesIdx_shuffled)))];
testIdx = [noIdx_shuffled(round(0.8*numel(noIdx_shuffled))+1:end); ...
           yesIdx_shuffled(round(0.8*numel(yesIdx_shuffled))+1:end)];

X_train = features(trainIdx, :);
y_train = numericLabels(trainIdx);

X_val = features(valIdx, :);
y_val = numericLabels(valIdx);

X_test = features(testIdx, :);
y_test = numericLabels(testIdx);

% Train the k-NN classifier
k = 5; % Number of nearest neighbors
mdl = fitcknn(X_train, y_train, 'NumNeighbors', k);

% Predict on the test set
y_pred = predict(mdl, X_test);

% Evaluate the classifier
accuracy = sum(y_pred == y_test) / numel(y_test);
disp(['Accuracy: ', num2str(accuracy * 100), '%']);

% Confusion matrix
figure;
cm = confusionchart(y_test, y_pred);
cm.Title = 'Confusion Matrix';

% Calculate precision, recall, and F1 score
precision = diag(cm.NormalizedValues) ./ sum(cm.NormalizedValues, 2);
recall = diag(cm.NormalizedValues) ./ sum(cm.NormalizedValues, 1)';
f1score = 2 * (precision .* recall) ./ (precision + recall);

fprintf('Precision: %.2f\n', mean(precision));
fprintf('Recall (Sensitivity): %.2f\n', mean(recall));
fprintf('F1 Score: %.2f\n', mean(f1score));
