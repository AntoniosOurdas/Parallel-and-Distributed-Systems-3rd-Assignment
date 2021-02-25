#include "utilities.h"

#define BILLION 1000000000L;

// struct timespec start_time;
struct timespec stop_time;

float calculateExecutionTime(struct timespec start_time)
{

    clock_gettime(CLOCK_MONOTONIC, &stop_time);

    float dSeconds = (stop_time.tv_sec - start_time.tv_sec);

    float dNanoSeconds = (float)(stop_time.tv_nsec - start_time.tv_nsec) / BILLION;

    return dSeconds + dNanoSeconds;
}

int main(int argc, char* argv[]) {

  // Read input arguments
  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  int w = atoi(argv[3]);

  float patchSigma = 2.0;
  sscanf(argv[4],"%f",&patchSigma);

  float filtSigma = 0.02;
  sscanf(argv[5],"%f",&filtSigma);

  // All arrays are 1 dimensional row major
  // Original Image extended to fit patches on the edges [(m+w-1)-by-(n+w-1)]
  float* X = (float*)malloc((m+w-1)*(n+w-1)*sizeof(float));

  // Filtered image [m-by-n]
  float* F = (float*)malloc(m*n*sizeof(float));

  // Stings used for input/output files
  char filename[30] = "";
  char filename2[30] = "";
  char filename3[30] = "";

  switch(m) {
    case 64:
      strcpy(filename, "lena_64");
      break;
    case 128:
      strcpy(filename, "lena_128");
      break;
    case 256:
      strcpy(filename, "lena_256");
      break;
    case 512:
      strcpy(filename, "lena_512");
      break;
  }

  strcpy(filename2, filename);
  strcpy(filename3, filename);

  // printf("Reading: %s\n", filename);
  // Read input image from txt file of floats
  char inputFile[50] = "../input_images/";
  strcat(inputFile, strcat(filename, ".txt"));
  printf("%s\n", inputFile);

  FILE* fptr = fopen(inputFile, "r");
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i)
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j)
      fscanf(fptr, "%f,", X+i*(n+w-1)+j);
      // X[i*(n+w-1)+j] = (i % (m / 8) == 0 || j % (n / 8) == 0) ? 1.0 : 0.0;
  fclose(fptr);

  // printf("figure;\n");
  // Print original image in matlab command format
  // printMatrixMatlab(X, m+w-1, n+w-1,"X");

  // Add gaussian noise to image (with standard deviation 0.04)
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i)
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j)
      X[i*(n+w-1)+j] += gaussianRand(0.04);


  // Fill edges mirroring the inside of image
  // similar to padarray(inputImage, [(w-1)/2 (w-1)/2], 'symmetric')

  // Right and left part
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i) {

    for(int j = 0; j < (w-1)/2; ++j) {
      X[i*(n+w-1)+j] = X[i*(n+w-1)+(w-j-2)];
    }

    for(int j = 1; j <= (w-1)/2 ; ++j) {
      X[i*(n+w-1)+((n+w-1)-j)] = X[i*(n+w-1)+((n+w-1)-(w-(j-1))+1)];
    }

  }

  // Upper and lower part
  for(int i = 0; i < m+(w-1); ++i) {

    for(int j = 0; j < (w-1)/2; ++j) {
      X[j*(n+w-1)+i] = X[(w-j-2)*(n+w-1)+i];
    }

    for(int j = 1; j <= (w-1)/2 ; ++j) {
      X[((n+w-1)-j)*(n+w-1)+i] = X[((n+w-1)-(w-(j-1))+1)*(n+w-1)+i];
    }

  }

  // Write noisy image to csv txt file
  // used by matlab script
  char outputFileNoisy[50] = "../output_images/";
  strcat(outputFileNoisy, strcat(filename2, "_noisy.txt"));
  printf("%s\n", outputFileNoisy);
  printMatrixCsv(X, m+w-1, n+w-1, outputFileNoisy);

  // Define and fill normalized gaussian patch window
  float* W = (float*)malloc(w*w*sizeof(float));
  float sum = 0.0;
  for(int i = 0; i < w; ++i) {
    for(int j = 0; j < w; ++j) {
      W[i*w+j] = exp((-pow((float)(i-w/2)/(float)w, 2)-pow((float)(j-w/2)/(float)w, 2))/(2.0*patchSigma));
      sum += W[i*w+j];
    }
  }

  for(int i = 0; i < w; ++i)
    for(int j = 0; j < w; ++j)
      W[i*w+j] /= sum;


  // Weight w(x,y)
  float Wxy = 0.0;
  // Sum of weights Z(i)
  float Zx = 0.0;
  // Distance of patches
  float D = 0.0;

  // Start measuring execution time
  struct timespec start_time;
	clock_gettime(CLOCK_MONOTONIC, &start_time);
  // Each pixel (i,j) of the denoised image
  // will be the weighted sum of
  // all other pixels of the noisy image
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i) {
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j) {
      // Set value of denoised image pixel
      // to 0 for summation
      F[(i-(w-1)/2)*n+(j-(w-1)/2)] = 0.0;
      // Then iterate all other pixels
      // and calulcate the weight Wxy
      // for each other pixel
      //
      // Zx is the normalization factor
      Zx = 0.0;
      for(int k = (w-1)/2; k < m+(w-1)/2; ++k) {
        for(int l = (w-1)/2; l < n+(w-1)/2; ++l) {
          // printf("\t(%d, %d):\n", k, l);
            Wxy = 0.0;
            D = 0.0;
            // Iterate all patch elements
            // to find similarity of patches around X(i,j) and X(k,l)
            // using gaussian kernel
            // (p,q) is patch pixel relative to the center pixel (0,0)
            for(int p = -(w-1)/2; p <= (w-1)/2; ++p) {
              for(int q = -(w-1)/2; q <= (w-1)/2; ++q) {
                D += W[(p+(w-1)/2)*w+(q+(w-1)/2)]*pow((X[(i+p)*(n+w-1)+(j+q)] - X[(k+p)*(n+w-1)+(l+q)]), 2.0);
              }
            }
            // Set weight which shows similarity
             // of patches (i,j) and (k,l)
            Wxy = exp(-D/(filtSigma*filtSigma));
            // Add weight to Z(i) for normalization
            Zx += Wxy;
            // Add weighted pixel to final sum
            F[(i-(w-1)/2)*n+(j-(w-1)/2)] += Wxy * X[k*(n+w-1)+l];
        }
      }
      // Normalize by dividing with Z(i)
      // This is our final denoised image pixel value
      // After this we continue to the next pixel
      F[(i-(w-1)/2)*n+(j-(w-1)/2)] /= Zx;
    }
  }

  // Write filtered image to csv txt file
  // used by matlab script
  char outputFileDenoised[30] = "../output_images/";
  strcat(outputFileDenoised, strcat(filename3, "_denoised.txt"));
  printf("%s\n", outputFileDenoised);
  printMatrixCsv(F, m, n, outputFileDenoised);

  printf("\n%% V0 run time: %f\n", calculateExecutionTime(start_time));

  return 0;
}
