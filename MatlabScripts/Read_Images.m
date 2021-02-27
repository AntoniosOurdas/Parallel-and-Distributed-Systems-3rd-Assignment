function result = Read_Images(size, patchSize, version, name)
close all
Lena = csvread(strcat(strcat('../input_images/',strcat(name,'_')), strcat(num2str(size), '.txt')));
Lena_noisy = csvread(strcat(strcat('../output_images/output_images_csv_txt/output_images_', version), strcat(strcat(strcat('/',strcat(name,'_')), strcat(num2str(size), strcat(strcat('_', strcat(num2str(patchSize), '_noisy.txt'))))))));
Lena_denoised = csvread(strcat(strcat('../output_images/output_images_csv_txt/output_images_', version), strcat(strcat('/',strcat(name,'_')), strcat(num2str(size), strcat(strcat('_', strcat(num2str(patchSize), '_denoised.txt')))))));
Lena_residual = csvread(strcat(strcat('../output_images/output_images_csv_txt/output_images_', version), strcat(strcat('/',strcat(name,'_')), strcat(num2str(size), strcat(strcat('_', strcat(num2str(patchSize), '_residual.txt')))))));

figure;

subplot(1,4,1);
imshow(Lena,[]);

title('Original');

subplot(1,4,2);
imshow(Lena_noisy,[]);
title('Noisy');
imwrite(Lena_noisy, strcat(strcat('../output_images/output_images_png_Matlab/output_images_png_', version), strcat(strcat('/',strcat(name,'_')), strcat(num2str(size), strcat(strcat('_', strcat(num2str(patchSize), '_noisy.png')))))));

subplot(1,4,3);
imshow(Lena_denoised,[]);
title('Denoised');
imwrite(Lena_denoised, strcat(strcat('../output_images/output_images_png_Matlab/output_images_png_', version), strcat(strcat('/',strcat(name,'_')), strcat(num2str(size), strcat(strcat('_', strcat(num2str(patchSize), '_denoised.png')))))));

subplot(1,4,4);
imshow(Lena_residual,[]);
title('Residual');
imwrite(Lena_residual, strcat(strcat('../output_images/output_images_png_Matlab/output_images_png_', version), strcat(strcat('/',strcat(name,'_')), strcat(num2str(size), strcat(strcat('_', strcat(num2str(patchSize), '_residual.png')))))));

saveas(gcf, strcat('../output_images/output_images_png_Matlab/output_images_figures_png', strcat(strcat('/',strcat(name,'_')), strcat(num2str(size), strcat(strcat('_', strcat(num2str(patchSize), strcat('_figure_', strcat(version, '.png')))))))));
result = 1;
end
