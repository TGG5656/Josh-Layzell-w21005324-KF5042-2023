clc; clear;
YOLOv3 = downloadPretrainedYOLOv3Detector();
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;
vehicleDataset.imageFilename = fullfile(pwd, vehicleDataset.imageFilename);
rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * length(shuffledIndices));
trainingDataTbl = vehicleDataset(shuffledIndices(1:idx), :);
testDataTbl = vehicleDataset(shuffledIndices(idx+1:end), :);
imdsTrain = imageDatastore(trainingDataTbl.imageFilename);
imdsTest = imageDatastore(testDataTbl.imageFilename);
bldsTrain = boxLabelDatastore(trainingDataTbl(:,2:end));
bldsTest = boxLabelDatastore(testDataTbl(:,2:end));
trainingData = combine(imdsTrain, bldsTrain);
testData = combine(imdsTest, bldsTest);
validateInputData(trainingData);
validateInputData(testData);
results = detect(YOLOv3,testData,'MiniBatchSize',8);
[ap,recall,precision] = evaluateDetectionPrecision(results,testData);
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
testImages = cellstr(testDataTbl.imageFilename);
for i = 1:min(5,numel(testImages))
    I = imread(testImages{i});
    [bboxes,scores,labels] = detect(YOLOv3, I);
    [bboxes,scores,labels] = selectStrongestBboxMulticlass(bboxes,scores,labels,...
        'RatioType','Min','OverlapThreshold', 0.5);
    if ~isempty(bboxes)
        I = insertObjectAnnotation(I, 'rectangle', bboxes, scores, 'LineWidth', 2, ...
            'FontSize', 12, 'Color', 'green', 'TextColor','black');
    end
    figure
    imshow(I)
end
