function result = Read_Images(size)
Lena = csvread(strcat('../input_images/lena_', strcat(num2str(size), '.txt')));
Lena_noisy = csvread(strcat('../output_images/lena_', strcat(num2str(size), '_noisy.txt')));
Lena_denoised = csvread(strcat('../output_images/lena_', strcat(num2str(size), '_denoised.txt')));

subplot(1,3,1);
imshow(Lena,[]);
title('Original');

subplot(1,3,2);
imshow(Lena_noisy,[]);
title('Noisy');

subplot(1,3,3);
imshow(Lena_denoised,[]);
title('Denoised');

result = 1;
end