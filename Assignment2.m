clc;
clear;
close all;

%% Part 1: Take 3 sets of 2 photographs to be stitched together to create a panorama

% You need to take photograph pairs to be stitched together, similar to the
% pair we talked about in our transformations lecture. We provide one set 
% of images for this assignment (namely S1-im1.png and S1-im2.png) and you 
% need to take extra 3 sets of your own. Take each pair of images from 
% different scenes. I would recommend taking many sets of images and 
% determining which ones to use after some experimentation with your 
% implementation and results. Make sure to resize your images to get the 
% longer dimension of the image (height or width) to be 750.
% The images you stitch together have to be different photographs of the 
% same scene. You can not crop an image in different places to stitch them 
% back together.
% For Part 5, at least two of your 3 image sets should contain 4 images to 
% be stitched together.
% 
% Submit your image pairs named S2-im1.png, S2-im2.png, and so on.

% S1
I1S1 = imread('S1-im1.png');
I1S1_gray = rgb2gray(I1S1); % convert the image to grayscale
I1S1_gray = im2double(I1S1_gray);

I2S1 = imread('S1-im2.png');
I2S1_gray = rgb2gray(I2S1); % convert the image to grayscale
I2S1_gray = im2double(I2S1_gray);

% S2
I1S2 = imread('S2-im1.png');
I1S2 = convert_to_750(I1S2);
I1S2_gray = rgb2gray(I1S2); % convert the image to grayscale
I1S2_gray = im2double(I1S2_gray);

I2S2 = imread('S2-im2.png');
I2S2 = convert_to_750(I2S2);
I2S2_gray = rgb2gray(I2S2); % convert the image to grayscale
I2S2_gray = im2double(I2S2_gray);

% S3
I1S3 = imread('S3-im1.png');
I1S3 = convert_to_750(I1S3);
I1S3_gray = rgb2gray(I1S3); % convert the image to grayscale
I1S3_gray = im2double(I1S3_gray);

I2S3 = imread('S3-im2.png');
I2S3 = convert_to_750(I2S3);
I2S3_gray = rgb2gray(I2S3); % convert the image to grayscale
I2S3_gray = im2double(I2S3_gray);

I3S3 = imread('S3-im3.png');
I3S3 = convert_to_750(I3S3);
I3S3_gray = rgb2gray(I3S3); % convert the image to grayscale
I3S3_gray = im2double(I3S3_gray);

I4S3 = imread('S3-im4.png');
I4S3 = convert_to_750(I4S3);
I4S3_gray = rgb2gray(I4S3); % convert the image to grayscale
I4S3_gray = im2double(I4S3_gray);

% S4
I1S4 = imread('S4-im1.png');
I1S4 = convert_to_750(I1S4);
I1S4_gray = rgb2gray(I1S4); % convert the image to grayscale
I1S4_gray = im2double(I1S4_gray);

I2S4 = imread('S4-im2.png');
I2S4 = convert_to_750(I2S4);
I2S4_gray = rgb2gray(I2S4); % convert the image to grayscale
I2S4_gray = im2double(I2S4_gray);

I3S4 = imread('S4-im3.png');
I3S4 = convert_to_750(I3S4);
I3S4_gray = rgb2gray(I3S4); % convert the image to grayscale
I3S4_gray = im2double(I3S4_gray);

I4S4 = imread('S4-im4.png');
I4S4 = convert_to_750(I4S4);
I4S4_gray = rgb2gray(I4S4); % convert the image to grayscale
I4S4_gray = im2double(I4S4_gray);

%% Part 2A: FAST feature detector

% Parameter
n = 12;
t = 0.2;

figure;
tic;
[yI1S1,xI1S1] = my_fast_detector(n, t, I1S1_gray);

FASTtime = toc;
imshow(I1S1);
hold on;
plot(xI1S1, yI1S1, 'r+', 'MarkerSize', 5);
saveas(gcf, 'S1-fast.png');

figure;
[yI1S2,xI1S2] = my_fast_detector(n, t, I1S2_gray);
imshow(I1S2);
hold on;
plot(xI1S2, yI1S2, 'r+', 'MarkerSize', 5);
saveas(gcf, 'S2-fast.png');
drawnow;
%% Part 2B: Robust FAST using Harris Cornerness metric (1 pts.)

% Parameter
w = 5;
r_t = 0.2;

figure;
tic;
[yI1S1r,xI1S1r] = my_fastr_detector(n, t, I1S1_gray, w, r_t);
FASTRtime = toc;
imshow(I1S1);
hold on;
plot(xI1S1r, yI1S1r, 'g+', 'MarkerSize', 5);
saveas(gcf, 'S1-fastr.png');

figure;
[yI1S2r,xI1S2r] = my_fastr_detector(n, t, I1S2_gray, w, r_t);
imshow(I1S2);
hold on;
plot(xI1S2r, yI1S2r, 'g+', 'MarkerSize', 5);
saveas(gcf, 'S2-fastr.png');

drawnow;
%% Part 3: Point description and matching

% FAST
%convert keypoints to a corner object for S1 images
fast_points_I1S1 = cornerPoints([xI1S1, yI1S1]);
[yI2S1,xI2S1] = my_fast_detector(n, t, I2S1_gray);
fast_points_I2S1 = cornerPoints([xI2S1, yI2S1]);

%convert keypoints to a corner object for S2 images
fast_points_I1S2 = cornerPoints([xI1S2, yI1S2]);
[yI2S2,xI2S2] = my_fast_detector(n, t, I2S2_gray);
fast_points_I2S2 = cornerPoints([xI2S2, yI2S2]);

%extract descriptors for S1 images
[featuresI1S1, validPointsI1S1] = extractFeatures(I1S1_gray, fast_points_I1S1);
[featuresI2S1, validPointsI2S1] = extractFeatures(I2S1_gray, fast_points_I2S1);

%extract descriptors for S2 images
[featuresI1S2, validPointsI1S2] = extractFeatures(I1S2_gray, fast_points_I1S2);
[featuresI2S2, validPointsI2S2] = extractFeatures(I2S2_gray, fast_points_I2S2);

%match features S1 & display
tic;
matchedPairsS1 = matchFeatures(featuresI1S1, featuresI2S1);
MPS1_time = toc;
matchedPointsI1S1 = validPointsI1S1(matchedPairsS1(:,1),:);
matchedPointsI2S1 = validPointsI2S1(matchedPairsS1(:,2),:);

figure;
showMatchedFeatures(I1S1, I2S1, matchedPointsI1S1, matchedPointsI2S1,"montage");
saveas(gcf, 'S1-fastMatch.png');

%match features S2 & display
matchedPairsS2 = matchFeatures(featuresI1S2, featuresI2S2);
matchedPointsI1S2 = validPointsI1S2(matchedPairsS2(:,1),:);
matchedPointsI2S2 = validPointsI2S2(matchedPairsS2(:,2),:);

figure;
showMatchedFeatures(I1S2, I2S2, matchedPointsI1S2, matchedPointsI2S2,"montage");
saveas(gcf, 'S2-fastMatch.png');

%FASTR
%convert keypoints to a corner object for S1 images
fastr_points_I1S1 = cornerPoints([xI1S1r, yI1S1r]);
[yI2S1r,xI2S1r] = my_fastr_detector(n, t, I2S1_gray, w, r_t);
fastr_points_I2S1 = cornerPoints([xI2S1r, yI2S1r]);

%convert keypoints to a corner object for S2 images
fastr_points_I1S2 = cornerPoints([xI1S2r, yI1S2r]);
[yI2S2r,xI2S2r] = my_fastr_detector(n, t, I2S2_gray, w, r_t);
fastr_points_I2S2 = cornerPoints([xI2S2r, yI2S2r]);

%extract descriptors for S1 images
[featuresI1S1r, validPointsI1S1r] = extractFeatures(I1S1_gray, fastr_points_I1S1);
[featuresI2S1r, validPointsI2S1r] = extractFeatures(I2S1_gray, fastr_points_I2S1);

%extract descriptors for S2 images
[featuresI1S2r, validPointsI1S2r] = extractFeatures(I1S2_gray, fastr_points_I1S2);
[featuresI2S2r, validPointsI2S2r] = extractFeatures(I2S2_gray, fastr_points_I2S2);

%match features S1 & display
tic;
matchedPairsS1r = matchFeatures(featuresI1S1r, featuresI2S1r);
MPS1r_time = toc;
matchedPointsI1S1r = validPointsI1S1r(matchedPairsS1r(:,1),:);
matchedPointsI2S1r = validPointsI2S1r(matchedPairsS1r(:,2),:);

figure;
showMatchedFeatures(I1S1, I2S1, matchedPointsI1S1r, matchedPointsI2S1r,"montage");
saveas(gcf, 'S1-fastRMatch.png');
%match features S2 & display
matchedPairsS2r = matchFeatures(featuresI1S2r, featuresI2S2r);
matchedPointsI1S2r = validPointsI1S2r(matchedPairsS2r(:,1),:);
matchedPointsI2S2r = validPointsI2S2r(matchedPairsS2r(:,2),:);

figure;
showMatchedFeatures(I1S2, I2S2, matchedPointsI1S2r, matchedPointsI2S2r,"montage");
saveas(gcf, 'S2-fastRMatch.png');

drawnow;
%% Part 4 RANSAC and Panoramas
% FAST Parameters
MaxNumTrials = 900;
Confidence = 99;
MaxDistance = 1.5;

% FASTR Parameters
MaxNumTrialsR = 1000;
ConfidenceR = 99;
MaxDistanceR = 5;

% Note: create_panoram contains the estgeotform2d function which takes in
% the corresponding parameters
%FAST
% Create S1 panorama
figure;
panorama_S1 = create_panorama({I1S1, I2S1}, MaxNumTrials, Confidence, MaxDistance, 'FAST', n, t, w, r_t);
imshow(panorama_S1);

% Create S2 panorama
figure;
panorama_S2 = create_panorama({I1S2, I2S2}, MaxNumTrials, Confidence, MaxDistance, 'FAST', n, t, w, r_t);
imshow(panorama_S2);

% Create S3 panorama
figure;
panorama_S3 = create_panorama({I1S3, I2S3, I3S3, I4S3}, MaxNumTrials, Confidence, MaxDistance, 'FAST', n, t, w, r_t);
imshow(panorama_S3);

% Create S4 panorama
figure;
panorama_S4 = create_panorama({I1S4, I2S4, I3S4, I4S4}, MaxNumTrials, Confidence, MaxDistance, 'FAST', n, t, w, r_t);
imshow(panorama_S4);

%FASTR
% Create S1 panorama
panorama_S1R = create_panorama({I1S1, I2S1}, MaxNumTrialsR, ConfidenceR, MaxDistanceR, 'FASTR', n, t, w, r_t);
panorama_S1R = convert_to_750(panorama_S1R);
imwrite(panorama_S1R, 'S1-panorama.png');

% Create S2 panorama
panorama_S2R = create_panorama({I1S2, I2S2}, MaxNumTrialsR, ConfidenceR, MaxDistanceR, 'FASTR', n, t, w, r_t);
panorama_S2R = convert_to_750(panorama_S2R);
imwrite(panorama_S2R, 'S2-panorama.png');

% Create S3 panorama
panorama_S3R = create_panorama({I1S3, I2S3, I3S3, I4S3}, MaxNumTrialsR, ConfidenceR, MaxDistanceR, 'FASTR', n, t, w, r_t);
panorama_S3R = convert_to_750(panorama_S3R);
imwrite(panorama_S3R, 'S3-panorama.png');

% Create S4 panorama
panorama_S4R = create_panorama({I1S4, I2S4, I3S4, I4S4}, MaxNumTrialsR, ConfidenceR, MaxDistanceR, 'FASTR', n, t, w, r_t);
panorama_S4R = convert_to_750(panorama_S4R);
imwrite(panorama_S4R, 'S4-panorama.png');