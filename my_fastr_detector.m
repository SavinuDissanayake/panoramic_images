function [ynew,xnew] = my_fastr_detector(n,t,inputImage,windowsize,r_threshold)
    [a, b] = my_fast_detector(n, t, inputImage);
    coords = [a, b];
    [height, width, ~] = size(inputImage);
    k = 0.05;
    ynew = zeros(length(coords),1);
    xnew = zeros(length(coords),1);
    count = 0;
    for i = 1:length(coords)
        y = coords(i,1);
        x = coords(i,2);
        halfSize = floor(windowsize/2);
        
        rowRange = max(1,(x - halfSize)):min((x + halfSize),width);
        colRange = max(1,(y - halfSize)):min((y + halfSize),height);

        window = inputImage(colRange, rowRange);
        [Ix, Iy] = gradient(window);
        temp = sum((Ix.*Iy),"all");
        M = [sum((Ix.^2),"all"), temp;
             temp, sum((Iy.^2),"all")];

        R = det(M)-k*(trace(M)^2);
        if R >= r_threshold
            count = count + 1;
            ynew(count) = y;
            xnew(count) = x;
        end
    end
    
    ynew = ynew(1:count);
    xnew = xnew(1:count);
end

