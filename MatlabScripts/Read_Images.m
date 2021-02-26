function result = Read_Images(size, patchSize)
Lena = csvread(strcat('../input_images/lena_', strcat(num2str(size), '.txt')));
Lena_noisy = csvread(strcat('../output_images/lena_', strcat(num2str(size), strcat(strcat('_', strcat(num2str(patchSize), '_noisy.txt'))))));
Lena_denoised = csvread(strcat('../output_images/lena_', strcat(num2str(size), strcat(strcat('_', strcat(num2str(patchSize), '_denoised.txt'))))));
Lena_residual = csvread(strcat('../output_images/lena_', strcat(num2str(size), strcat(strcat('_', strcat(num2str(patchSize), '_residual.txt'))))));

figure;

subplot(1,4,1);
imshow(Lena,[]);
title('Original');

subplot(1,4,2);
imshow(Lena_noisy,[]);
title('Noisy');
imwrite(Lena_noisy, strcat(strcat('../output_images_png/lena_noisy_n_', num2str(size)), strcat('_w_', strcat(num2str(patchSize),  '.png'))));

subplot(1,4,3);
imshow(Lena_denoised,[]);
title('Denoised');
imwrite(Lena_denoised, strcat(strcat('../output_images_png/lena_denoised_n_', num2str(size)), strcat('_w_', strcat(num2str(patchSize),  '.png'))));

subplot(1,4,4);
imshow(Lena_residual,[]);
title('Residual');
imwrite(Lena_residual, strcat(strcat('../output_images_png/lena_residual_n_', num2str(size)), strcat('_w_', strcat(num2str(patchSize),  '.png'))));

saveas(gcf, strcat('../output_images_png/lena_n_', strcat(num2str(size), strcat('_w_', strcat(num2str(patchSize),  '.png')))));
result = 1;
end
