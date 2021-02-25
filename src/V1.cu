#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

// CUDA kernel to compute denoise image
__global__ void nonLocalMeans(double* P, int m, int n, int w, double filtSigma, double* F) {

  double Wxy = 0.0;
  double Zx = 0.0;
  double D = 0.0;

  int i = blockIdx.x;
  int j = threadIdx.x;

  F[i*n+j] = 0.0;
  Zx = 0.0;
  for(int k = 0; k < m; ++k) {
    for(int l = 0; l < n; ++l) {
        Wxy = 0.0;
        D = 0.0;
        for(int p = -(w-1)/2; p <= (w-1)/2; ++p) {
          for(int q = -(w-1)/2; q <= (w-1)/2; ++q) {
            // D += pow(P[(i+p)*(n+w-1)+(j+q)] - P[(k+p)*(n+w-1)+(l+q)], 2.0);
            D += pow((P[i*n*w*w+j*w*w+(p+(w-1)/2)*w+(q+(w-1)/2)] - P[k*n*w*w+l*w*w+(p+(w-1)/2)*w+(q+(w-1)/2)]), 2.0);

          }
        }
        Wxy = exp(-D/(filtSigma*filtSigma));
        Zx += Wxy;
        // P[k][l][(w-1)/2][(w-1)/2] is the center pixel of current patch
        F[i*n+j] += Wxy * P[k*n*w*w + l*w*w + (w-1)/2*w + (w-1)/2];
    }
  }
  F[i*n+j] /= Zx;

  return;
}

// Auxiliary matrix printing functions
void printMatrixInt(int* A, int n, int m) {
  for(int j = 0; j < m*2+3; ++j) {
    printf("-");
  }
  printf("\n");
  for(int i = 0; i < n; ++i) {
    printf("| ");
    for(int j = 0; j < m; ++j) {
      printf("%d ", A[i*m+j]);
    }
    printf("|\n");
  }
  for(int j = 0; j < m*2+3; ++j) {
    printf("-");
  }
  printf("\n");
}

void printMatrixDouble(double* A, int n, int m) {
  for(int j = 0; j < m*11; ++j) {
    printf("-");
  }
  printf("\n");
  for(int i = 0; i < n; ++i) {
    printf("| ");
    for(int j = 0; j < m; ++j) {
      if(A[i*m+j] < 0.0)
        printf("%lf ", A[i*m+j]);
      else
        printf(" %lf ", A[i*m+j]);
    }
    printf(" |\n");
  }
  for(int j = 0; j < m*11; ++j) {
    printf("-");
  }
  printf("\n");
}

void printMatrixMatlab(double* A, int m, int n, char* name) {
  printf("%s = [", name);
  for(int i = 0; i < m; ++i) {
    for(int j = 0; j < n; ++j)
      printf("%lf ", A[i*n+j]);
    printf(";");
  }
  printf("];\n");
}

void printMatrixMatlab3D(double* A, int d1, int d2, int d3, char* name) {
  for(int i = 0; i < d1; ++i) {
    printf("%s(:,:,%d) = [", name, i+1);
    for(int j = 0; j <d2; ++j) {
      for(int k = 0; k < d3; ++k)
        printf("%lf ", A[i*d2*d3+j*d3+k]);
      printf(";");
    }
    printf("];\n");
  }
}

// Auxiliary function to generate random gaussian number
// used for adding gaussian noise
double gaussianRand(double sigma) {
  double sum = 0.0;
  for(int i = 0; i < 30; ++i) {
    sum += -2.0*sqrtf(12)*sigma*(double)rand()/RAND_MAX+1.0*sqrtf(12)*sigma;
  }
  sum /= 30.0;
  return sum;
}

// Function for writing image array to csv format txt file
// which can then be read by ReadImage.m Matlab script
void printMatrixCsv(double* A, int m, int n, char* name) {
  FILE* ptr = fopen(name, "w");
  if(ptr == NULL) {
    printf("Failed to create output file. Exiting!\n");
    return;
  }
  for(int i = 0; i < m; ++i) {
    fprintf(ptr, "%f", A[i*n]);
    for(int j = 0; j < n; ++j) {
      fprintf(ptr, ",%lf", A[i*n+j]);
    }
    fprintf(ptr,"\n");
  }
  fclose(ptr);
}

// Main function
int main(int argc, char* argv[]) {

  // Read input arguments
  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  int w = atoi(argv[3]);

  double patchSigma = 2.0;
  sscanf(argv[4],"%lf",&patchSigma);

  double filtSigma = 0.02;
  sscanf(argv[5],"%lf",&filtSigma);

  // Create gaussian kernel
  double* W = (double*)malloc(w*w*sizeof(double));
  double sum = 0.0;
  for(int i = 0; i < w; ++i) {
    for(int j = 0; j < w; ++j) {
      W[i*w+j] = exp((-pow((double)(i-w/2)/(double)w, 2)-pow((double)(j-w/2)/(double)w, 2))/(2.0*patchSigma*patchSigma));
      sum += W[i*w+j];
    }
  }

  // Normalize
  for(int i = 0; i < w; ++i)
    for(int j = 0; j < w; ++j)
      W[i*w+j] /= sum;


  // Original Image extended to fit patches on the edges [(m+w-1)-by-(n+w-1)]
  double* X = (double*)malloc((m+w-1)*(n+w-1)*sizeof(double));
  // 3D Patch Cube [m-by-n-by-w-by-w]
  double* P = (double*)malloc(m*n*w*w*sizeof(double));
  // Filtered image [m-by-n]
  double* F = (double*)malloc(m*n*sizeof(double));

  // 3D Patch cude pointer for GPU memory
  double* deviceP = NULL;
  cudaMalloc(&deviceP, m*n*w*w*sizeof(double));

  // Filtered image pointer for GPU memory
  double* deviceF = NULL;
  cudaMalloc(&deviceF, m*n*sizeof(double));

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

  char inputFile[50] = "../input_images/";
  strcat(inputFile, strcat(filename, ".txt"));
  printf("%s\n", inputFile);

  // Read input image
  FILE* fptr = fopen(inputFile, "r");
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i)
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j)
      fscanf(fptr, "%lf,", X+i*(n+w-1)+j);
      // X[i*(n+w-1)+j] = (i+j);
  fclose(fptr);

  // Add noise to input image
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i)
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j)
      X[i*(n+w-1)+j] += gaussianRand(0.1);


  // Fill edges mirroring the inside of image
  // similar to padarray(inputImage, [(w-1)/2 (w-1)/2], 'symmetric')

  // Right and left part
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i) {

    for(int j = 0; j < (w-1)/2; ++j) {
      X[i*(n+w-1)+j] = X[i*(n+w-1)+(w-j-2)];
    }

    for(int j = 1; j <= (w-1)/2 ; ++j) {
      X[i*(n+w-1)+((n+w-1)-j)] = X[i*(n+w-1)+((n+w-1)-(w-(j-1)-1))];
    }

  }

  // Upper and lower part
  for(int i = 0; i < m+(w-1); ++i) {

    for(int j = 0; j < (w-1)/2; ++j) {
      X[j*(n+w-1)+i] = X[(w-j-2)*(n+w-1)+i];
    }

    for(int j = 1; j <= (w-1)/2 ; ++j) {
      X[((n+w-1)-j)*(n+w-1)+i] = X[((n+w-1)-(w-(j-1)-1))*(n+w-1)+i];
    }

  }

  // Calculate all w-by-w patches from X multiplied
  // with gaussian kernel and save them to P
  // (i,j) is the center pixel of each patch
  // (k,l) is the patch element
  // appropriate offsets are used
  for(int i = (w-1)/2; i < m+(w-1)/2;++i) {
      for(int j = (w-1)/2; j < n+(w-1)/2; ++j) {
        for(int k = -(w-1)/2; k <= (w-1)/2; ++k) {
            for(int l = -(w-1)/2; l <= (w-1)/2; ++l) {
              P[(i-(w-1)/2)*n*w*w+(j-(w-1)/2)*w*w+(k+(w-1)/2)*w+(l+(w-1)/2)] =
              X[(i+k)*(n+w-1)+(j+l)]*W[(k+(w-1)/2)*w+(l+(w-1)/2)];
            }
        }
      }
  }

  // Write noisy image to csv txt file
  // used by matlab script
  char outputFileNoisy[50] = "../output_images/";
  strcat(outputFileNoisy, strcat(filename2, "_noisy.txt"));
  printf("%s\n", outputFileNoisy);
  printMatrixCsv(X, m+w-1, n+w-1, outputFileNoisy);

  // Copy data for input and output
  // from CPU memory to GPU memory
  cudaMemcpy(deviceP, P, m*n*w*w*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceF, F, m*n*sizeof(double), cudaMemcpyHostToDevice);

  // CUDA events used for measuring time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Start measuring time and call kernel
  cudaEventRecord(start);
  nonLocalMeans<<<m,n>>>(deviceP, m, n, w, filtSigma, deviceF);
  cudaEventRecord(stop);

  // Copy data for input and output
  // from CPU memory to GPU memory
  cudaMemcpy(P, deviceP, m*n*w*w*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(F, deviceF, m*n*sizeof(double), cudaMemcpyDeviceToHost);

  // Find original denoised pixel
  // divinding by center pixel
  // of gaussian kernel value
  for(int i = 0; i < m; ++i)
    for(int j = 0; j < n; ++j)
      F[i*n+j] /= W[(w-1)/2*w+(w-1)/2];

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Write filtered image to csv txt file
  // used by matlab script
  char outputFileDenoised[30] = "../output_images/";
  strcat(outputFileDenoised, strcat(filename3, "_denoised.txt"));
  printf("%s\n", outputFileDenoised);
  printMatrixCsv(F, m, n, outputFileDenoised);

  printf("%%Exceution time: %lf\n", milliseconds);

  // Deallocate CPU and GPU memory
  cudaFree(deviceP);
  cudaFree(deviceF);
  free(X);
  free(F);
  free(W);
  free(P);

  return 0;
}
