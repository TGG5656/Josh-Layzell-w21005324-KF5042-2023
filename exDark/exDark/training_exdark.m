% Set the directory where the ExDark dataset is located
exDarkDir = 'C:\Users\joshu\OneDrive\Documents\MATLAB\exDark';

% Define the folder names for the images and annotations
imgFolder = fullfile(pwd, 'exDark', 'images');
annotFolder = fullfile(pwd, 'exDark', 'annotations');

trainingData = objectDetectorTrainingData(imgFolder, annotFolder);
blds = boxLabelDatastore(trainingData(:,2:end));
imds = imageDatastore(trainingData.imageFilename);

% Combine the imageDatastore and boxLabelDatastore into a training dataset
trainingData = combine(imds, blds);

% Split the training dataset into training and testing datasets
[trainData, testData] = splitEachLabel(trainingData, 0.8, 'randomized');
