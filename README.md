# Parallel-and-Distributed-Systems-3rd-Assignment
3rd Assignment for Parallel and Distibuted course of Aristotle University of Thessaloniki

### Project directories
1. **input_images** contains images as text files in csv format given as input to the program
2. **MatlabScript** contains Matlab script used for reading output images and creating figures showing original image, image with additive gaussian noise, denoised image and residual image (difference of filtered and noisy image)
3. **output_images** contains output images as text files in csv format given as input to the program
4. **output_images_png** contains output images as png after matlab proccessing
5. **src** contains source code including V0, V1, V2, utilities and Makefile

### Compilation and usage of program
To compile the program simply go to src directory use provided Makefile by typing make all.

Usage of program is show below

```
Usage: V0 m n w input_image
  m:            height of input image
  n:            width of input image
  w:            patch window size
  input_image:  input image (text file in csv format only)
```
