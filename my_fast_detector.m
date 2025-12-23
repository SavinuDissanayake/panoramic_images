function [y,x] = my_fast_detector(n,t,inputImage)
%MY_FAST_DETECTOR Takes in 2 thresholds, n and t, and an input image.
%Outputs an image with the features drawn on
%   t = pixel intensity threshold
%   n = # of contiguous pixels
%   [y,x] = locations of detected pixels

% pad the image

[height, width, ~] = size(inputImage);
padded_image = padarray(inputImage, [3,3], "replicate");

bres_circle_3 = [3,  0; 3,  1; 2,  2; 1,  3;
                 0,  3; -1,  3; -2,  2; -3,  1;
                -3,  0; -3, -1; -2, -2; -1, -3;
                 0, -3; 1, -3; 2, -2; 3, -1];

% Create a matrix of zeros where each element is a 16-element array of zeros
pixel_light = zeros(height, width, 16);
pixel_dark = zeros(height, width, 16);
pixel_diffs = zeros(height, width, 16);

for idx = 1:size(bres_circle_3, 1)
    % Translate and crop the image
    i = bres_circle_3(idx, :);
    translated_image = imtranslate(padded_image, i);
    cropped_image = imcrop(translated_image, [4, 4, width-1, height-1]);

    % Determine which pixels are above or below the thresholds
    pixel_light_temp = inputImage > (cropped_image + t);
    pixel_dark_temp = inputImage < (cropped_image - t);
    pixel_diffs_temp = inputImage - cropped_image;
    
    % Update the pixel_light and pixel_dark arrays
    pixel_light(:, :, idx) = pixel_light_temp;
    pixel_dark(:, :, idx) = pixel_dark_temp;
    pixel_diffs(:,:, idx) = pixel_diffs_temp;
end

res_light = zeros(height, width);
res_dark = zeros(height, width);
v_score = zeros(height, width);
for i = 1:height
    for j = 1:width
        light_array_temp = zeros(1,16);
        dark_array_temp = zeros(1,16);
        diffs_array_temp = zeros(1,16);
        for k = 1:16
            light_array_temp(k) = pixel_light(i,j,k);
            dark_array_temp(k) = pixel_dark(i,j,k);
            diffs_array_temp(k) = pixel_diffs(i,j,k);
        end
        res_light(i,j) = detect_contiguous(light_array_temp, n);
        res_dark(i,j) = detect_contiguous(dark_array_temp, n);
        v_score(i,j) = sum(diffs_array_temp);
    end
end
res = (res_light | res_dark);
res2 = res .* v_score;
res3 = padarray(res2, [1,1], 0);
for i = 2:height
    for j = 2:width
        window = res3(i-1:i+1, j-1:j+1);
        if sum(window(:)) > 1
            [maxVal, linInd] = max(window(:));
            window = zeros(3,3);
            window(linInd) = maxVal;
            res3(i-1:i+1, j-1:j+1) = window;
        end
    end
end
[y,x] = find(res3(2:end-1,2:end-1));
return;
end

function outp = detect_contiguous(array, n)
    arr = [array, array];
    for ind = 1:length(array)
        if sum(arr(ind:ind+n-1)) == n
            outp = true;
            return;
        end
    end
    outp = false;
    return;
end