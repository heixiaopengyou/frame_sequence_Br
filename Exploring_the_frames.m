% Loading a sample frame
data = load('.../Briareo_dataset_mat30/train.zip/000/g11/00/tof/depth/022_z.mat');
% Extracting the frame
frame = data.arr_0;
frame1 = mapminmax(frame);
% frame1 = normalize(frame); % Creates NaN values in the frame (surely from dividing with 0) so I chose the MinMax scaler
% Resizing the frame to 8x8 using bicubic interpolation
resampled_frame = imresize(frame1, [8, 8], 'bicubic');
resampled_frame = cast(resampled_frame, 'single'); % Conversion from double to single

% Visualizing the original and resampled frame
figure;
subplot(1, 2, 1);
imshow(frame);
title('Original Frame (171x224)');
subplot(1, 2, 2);
imshow(resampled_frame);
title('Resampled Frame (8x8)');
