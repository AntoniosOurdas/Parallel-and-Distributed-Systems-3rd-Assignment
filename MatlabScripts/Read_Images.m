function result = Read_Images(size, patchSize)
Lena = csvread(strcat('../input_images/lena_', strcat(num2str(size), '.txt')));
Lena_noisy = csvread(strcat('../output_images/lena_', strcat(num2str(size), strcat(strcat('_', strcat(num2str(patchSize), '_noisy.txt'))))));
Lena_denoised = csvread(strcat('../output_images/lena_', strcat(num2str(size), strcat(strcat('_', strcat(num2str(patchSize), '_denoised.txt'))))));

figure;

subplot(1,3,1);
imshow(Lena,[]);
title('Original');

subplot(1,3,2);
imshow(Lena_noisy,[]);
title('Noisy');

subplot(1,3,3);
imshow(Lena_denoised,[]);
title('Denoised');

saveas(gcf, strcat('../output_images_png/lena_n_', strcat(num2str(size), strcat('_w_', strcat(num2str(patchSize),  '.png')))));
result = 1;
end
