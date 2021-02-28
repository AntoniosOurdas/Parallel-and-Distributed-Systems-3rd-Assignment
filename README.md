# Parallel-and-Distributed-Systems-3rd-Assignment
3rd Assignment for Parallel and Distibuted course of Aristotle University of Thessaloniki

### Project directories
1. **Execution_times** contains execution times and generated diagrams
2. **MatlabScript** contains Matlab script used for reading output images and creating figures showing original image, image with additive gaussian noise, denoised image and residual image (difference of filtered and noisy image)
3. **input_images** contains images as text files in csv format given as input to the program
4. **output_images** contains output images as text files both in csv format and png
5. **src** contains source code including V0, V1, V2, utilities and Makefile

### Compilation and usage of program
To compile the program simply go to src directory use provided Makefile by typing make all.

Usage of program is show below

```
Usage: ./V0 m n w filtSigma patchSigma filtSigma input_image output_image_name
  m:                 height of input image
  n:                 width of input image
  w:                 patch window size
  patchSigma:        standard deviation of gaussian kernel
  filtSigma:         standard deviation of filter
  input_image:       input image (text file in csv format only)
  output_image_name: name for ouput image
```
