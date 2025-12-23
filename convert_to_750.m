function newImg = convert_to_750(img)
    [height, width, ~] = size(img);

    if height > width
        scale = 750 / height;
    else
        scale = 750 / width;
    end

    newImg = imresize(img, scale);
end

