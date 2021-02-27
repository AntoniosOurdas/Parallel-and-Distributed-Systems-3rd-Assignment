#include "utilities.h"

#define BILLION 1000000000L;

// struct timespec start_time;
struct timespec stop_time;

double calculateExecutionTime(struct timespec start_time)
{

    clock_gettime(CLOCK_MONOTONIC, &stop_time);

    double dSeconds = (stop_time.tv_sec - start_time.tv_sec);

    double dNanoSeconds = (double)(stop_time.tv_nsec - start_time.tv_nsec) / BILLION;

    return dSeconds + dNanoSeconds;
}

int main(int argc, char* argv[]) {

  // Various checks for valid input arguments
  if(argc < 8) {
    printf("Usage: ./V0 m n w input_image output_image_name\n");
    return 1;
  }

  // Read input arguments
  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  int w = atoi(argv[3]);

  if(m != n) {
    printf("Only square images supported\n");
    return 1;
  }

  if(m != 64 && m != 128 && m != 256) {
    printf("Only 64x64, 128x128 and 256x256 size images supported\n");
    return 1;
  }

  if(w != 3 && w != 5 && w != 7) {
    printf("Only 3x3, 5x5 and 7x7 patch sizes supported\n");
    return 1;
  }

  double patchSigma = 2.0;
  sscanf(argv[4],"%lf",&patchSigma);

  double filtSigma = 0.02;
  sscanf(argv[5],"%lf",&filtSigma);

  // All arrays are 1 dimensional row major
  // Original Image extended to fit patches on the edges [(m+w-1)-by-(n+w-1)]
  double* X = (double*)malloc((m+w-1)*(n+w-1)*sizeof(double));

  // Filtered image [m-by-n]
  double* F = (double*)malloc(m*n*sizeof(double));

  // Residual image (F - x)
  double* R = (double*)malloc(m*n*sizeof(double));

  FILE* fptr = fopen(argv[6], "r");
  if(fptr == NULL) {
    printf("Error reading input image\n");
    return 1;
  }

  for(int i = (w-1)/2; i < m+(w-1)/2; ++i)
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j)
      fscanf(fptr, "%lf,", X+i*(n+w-1)+j);
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
  char outputFileName[100] = "";
  sprintf(outputFileName, "../output_images/output_images_csv_txt/output_images_V0/%s_%d_%d_noisy.txt", argv[7], n, w);
  // printf("Writing noisy image to %s\n", outputFileName);
  printMatrixCsv(X, m+w-1, n+w-1, outputFileName);

  // Define and fill normalized gaussian patch window
  double* W = (double*)malloc(w*w*sizeof(double));
  double sum = 0.0;
  for(int i = 0; i < w; ++i) {
    for(int j = 0; j < w; ++j) {
      W[i*w+j] = exp((-pow((double)(i-w/2)/(double)w, 2)-pow((double)(j-w/2)/(double)w, 2))/(2.0*patchSigma));
      sum += W[i*w+j];
    }
  }

  for(int i = 0; i < w; ++i)
    for(int j = 0; j < w; ++j)
      W[i*w+j] /= sum;


  // Weight w(x,y)
  double Wxy = 0.0;
  // Sum of weights Z(i)
  double Zx = 0.0;
  // Distance of patches
  double D = 0.0;

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

  // Calculate residual image
  for(int i = 0; i < m; ++i) {
    for(int j = 0; j < n; ++j) {
      R[i*n+j] = F[i*n+j] - X[(i+(w-1)/2)*(n+w-1)+(j+(w-1)/2)];
    }
  }

  // Write filtered and residual image to csv txt files
  // used by matlab script
  sprintf(outputFileName, "../output_images/output_images_csv_txt/output_images_V0/%s_%d_%d_denoised.txt", argv[7], n, w);
  // printf("Writing denoised image to %s\n", outputFileName);
  printMatrixCsv(F, m, n, outputFileName);

  sprintf(outputFileName, "../output_images/output_images_csv_txt/output_images_V0/%s_%d_%d_residual.txt", argv[7], n, w);
  // printf("Writing residual image to %s\n", outputFileName);
  printMatrixCsv(R, m, n, outputFileName);

  printf("%lf\n", calculateExecutionTime(start_time));

  free(X);
  free(W);
  free(F);
  free(R);
  return 0;
}
