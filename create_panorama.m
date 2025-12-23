function panorama = create_panorama(img_set,MaxNumTrials,Confidence,MaxDistance,method, nc, t, w, r_t)
    % Read the first image from the image set.
    I = img_set{1};
    
    % Initialize features for the first image.
    grayImage = im2gray(I);
    grayImage = im2double(grayImage);
    if strcmp(method, 'FAST')
        [y,x] = my_fast_detector(nc, t, grayImage);
    elseif strcmp(method, 'FASTR')
        [y,x] = my_fastr_detector(nc, t, grayImage, w, r_t);
    end
    points = cornerPoints([x,y]);
    [features,points] = extractFeatures(grayImage,points);

    % Initialize all the transformations to the identity matrix
    numImages = numel(img_set);
    tforms(numImages) = projtform2d;

    % Initialize variable to hold image sizes.
    imageSize = zeros(numImages,2);

    % Iterate over remaining image pairs
    for n = 2:numImages
        pointsPrevious = points;
        featuresPrevious = features;
            
        I = img_set{n};
        grayImage = im2gray(I);
        grayImage = im2double(grayImage);
        imageSize(n,:) = size(grayImage);
        if strcmp(method, 'FAST')
            [y,x] = my_fast_detector(nc, t, grayImage);
        elseif strcmp(method, 'FASTR')
            [y,x] = my_fastr_detector(nc, t, grayImage, w, r_t);
        end
        points = cornerPoints([x,y]);
        [features,points] = extractFeatures(grayImage,points);
      
        indexPairs = matchFeatures(features,featuresPrevious,Unique=true);
        matchedPoints = points(indexPairs(:,1), :);
        matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);        
        
        try
            tforms(n) = estgeotform2d(matchedPoints, matchedPointsPrev,"projective",'Confidence',Confidence,'MaxNumTrials',MaxNumTrials, 'MaxDistance', MaxDistance);
        catch ME
            n
            matchedPointsPrev
            matchedPoints
            pointsPrevious
            points
            featuresPrevious
            features
            figure;
            imshow(I)
            hold on;
            plot(x, y, 'g+', 'MarkerSize', 5);
            disp(ME.message())
        end
        tforms(n).A = tforms(n-1).A * tforms(n).A; 
    end
    
    % Create an initial, empty, panorama
    for idx = 1:numel(tforms)           
        [xlim(idx,:),ylim(idx,:)] = outputLimits(tforms(idx),[1 imageSize(idx,2)],[1 imageSize(idx,1)]);
    end
    maxImageSize = max(imageSize);

    % Find the minimum and maximum output limits.
    xMin = min([1; xlim(:)]);
    xMax = max([maxImageSize(2); xlim(:)]);
    
    yMin = min([1; ylim(:)]);
    yMax = max([maxImageSize(1); ylim(:)]);
    
    % Compute the width and height of the panorama.
    width  = round(xMax - xMin);
    height = round(yMax - yMin);

    % Initialize the panorama with a blank canvas
    panorama = zeros([height width 3],"like",I);
    
    % Create a 2-D spatial reference object to define the size of the panorama.
    xLimits = [xMin xMax];
    yLimits = [yMin yMax];
    panoramaView = imref2d([height width],xLimits,yLimits);
    
    % Create the panorama by warping each image to transform it into the panorama
    for idx = 1:numImages
        I = img_set{idx};   
        warpedImage = imwarp(I,tforms(idx),OutputView=panoramaView);                 
        mask = imwarp(true(size(I,1),size(I,2)),tforms(idx),OutputView=panoramaView);
        panorama = imblend(warpedImage,panorama,mask,foregroundopacity=1);
    end
end

