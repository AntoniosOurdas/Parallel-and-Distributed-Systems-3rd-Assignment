#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define BILLION 1000000000L;

// struct timespec start_time;
struct timespec stop_time;

__global__ double calculateExecutionTime(struct timespec start_time)
{

    clock_gettime(CLOCK_MONOTONIC, &stop_time);

    double dSeconds = (stop_time.tv_sec - start_time.tv_sec);

    double dNanoSeconds = (double)(stop_time.tv_nsec - start_time.tv_nsec) / BILLION;

    return dSeconds + dNanoSeconds;
}

__global__ void nonLocalMeans(double* X, int m, int n, int w, double filtSigma, double patchSigma, double* F) {

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

  // Print patch window in matlab command format
  // printMatrixMatlab(W, w, w, "w");
  // printMatrixDouble(W, w, w);

  // Weight w(x,y)
  double Wxy = 0.0;
  // Sum of weights Z(i)
  double Zx = 0.0;
  // Distance of patches
  double D = 0.0;

  // Start measuring execution time
  // struct timespec start_time;
  // clock_gettime(CLOCK_MONOTONIC, &start_time);
  // Each pixel (i,j) of the denoised image
  // will be the weighted sum of
  // all other pixels of the noisy image
  int i = blockIdx.x + (w-1)/2;
  int j = threadIdx.x + (w-1)/2;
  // for(int i = (w-1)/2; i < m+(w-1)/2; ++i) {
    // for(int j = (w-1)/2; j < n+(w-1)/2; ++j) {
      // Set value of denoised image pixel
      // to 0 for summation
      F[i*n+j] = 0.0;
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
                D += W[(p+(w-1)/2)*w+(q+(w-1)/2)]*pow(X[(i+p)*(n+w-1)+(j+q)] - X[(k+p)*(n+w-1)+(l+q)], 2.0);
                // printf("Distance: %lf\n", D);
              }
            }
            // Set weight which shows similarity
             // of patches (i,j) and (k,l)
            Wxy = exp(-D/(filtSigma*filtSigma));
            // Add weight to Z(i) for normalization
            Zx += Wxy;
            // Add weighted pixel to final sum
            F[i*n+j] += Wxy * X[k*n+l];
        }
      }
      // printf("Done: %lf\n", Zx);
      // Normalize sum
      // printf("%lf\n", F[i*n+j]);
      // Normalize by dividing with Z(i)
      // This is our final denoised image pixel value
      // After this we continue to the next pixel
      F[i*n+j] /= Zx;
      // printf("Done: %lf\n", F[i*n+j]);
    // }
  // }

  return;
}

__global__ void printMatrixInt(int* A, int n, int m) {
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

__global__ void printMatrixDouble(double* A, int n, int m) {
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

__global__ void printMatrixMatlab(double* A, int m, int n, char* name) {
  printf("%s = [", name);
  for(int i = 0; i < m; ++i) {
    for(int j = 0; j < n; ++j)
      printf("%lf ", A[i*n+j]);
    printf(";");
  }
  printf("];\n");
}

__device__ double gaussianRand(double sigma) {
  double sum = 0.0;
  for(int i = 0; i < 10; ++i) {
    sum += -2.0*sqrt(12)*sigma*(double)rand()/RAND_MAX+1.0*sqrt(12)*sigma;
  }
  sum /= 10.0;
  return sum;
}

int main(int argc, char* argv[]) {

  // Get image size
  int m = 5;
  int n = 5;

  // Get patch size
  int w = 1;

  // Image 1D array (row major) (extended to fit patches)
  double* X = (double*)malloc((m+w-1)*(n+w-1)*sizeof(double));

  double* deviceX = NULL;
  cudaMalloc(&deviceX, (m+w-1)*(n+w-1)*sizeof(double));
  // Read input image from txt file of doubles
  // FILE* fptr = fopen("myFile.txt", "r");

  for(int i = (w-1)/2; i < m+(w-1)/2; ++i)
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j)
      // fscanf(fptr, "%lf ", X+i*(n+w-1)+j);
      X[i*(n+w-1)+j] = (i % (m / 8) == 0 || j % (n / 8) == 0) ? 1.0 : 0.0;
  // fclose(fptr);

  printf("figure;\n");
  // Print original image in matlab command format
  printMatrixMatlab(X, m+w-1, n+w-1,"X");

  // Add gaussian noise to image (with standard deviation 0.05)
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i)
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j)
      X[i*(n+w-1)+j] += gaussianRand(0.1);


  // Fill edges mirroring the inside of image
  // Right and left part
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i) {

    for(int j = 0; j < (w-1)/2; ++j) {
      X[i*(n+w-1)+j] = X[i*(n+w-1)+((w-1)/2-j+1)];
    }

    for(int j = n+(w-1)/2; j < (n+w-1); ++j) {
      X[i*(n+w-1)+j] = X[i*(n+w-1)+((w-1)/2-(n+w-1-j-1))];
    }

  }

  // Print noisy image in matlab command format
  // printMatrixDouble(X, m+w-1, n+w-1);
  printMatrixMatlab(X, m+w-1, n+w-1, "Xn");

  // Define denoised image
  double* F = (double*)malloc(m*n*sizeof(double));

  // Get gaussian patch standard deviation
  double patchSigma = 1.0;

  cudaMemcpy(X, deviceX, m*n*sizeof(double), cudaMemcpyDeviceToHost);

  nonLocalMeans<<<m,n>>>(X, m, n, w, sigma, patchSigma, F);


  // printMatrixDouble(F, m, n);
  // printMatrixMatlab(F, m, n, "F");

  // printf("subplot(1,3,1);imshow(X,[]);title('Original');subplot(1,3,2);imshow(Xn,[]);title('Noisy');subplot(1,3,3);imshow(F,[]);title('Denoised');\n");
  // printf("\nV0 run time: %f\n", calculateExecutionTime(start_time));
  return 0;
}
