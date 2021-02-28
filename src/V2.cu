#include "utilities.cuh"

// CUDA kernel to compute denoise image
__global__ void nonLocalMeans(double* P, int m, int n, int w, double filtSigma, double* F) {

  __shared__ int size = (1024+(w-1))(1024+(w-1));
  __shared__ double localImageBlock[size];
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

// Main function
int main(int argc, char* argv[]) {

  // Various checks for valid input arguments
  if(argc < 8) {
    printf("Usage: ./V1 m n w input_image output_image_name\n");
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
  // Residual image (F - x)
  double* R = (double*)malloc(m*n*sizeof(double));

  // 3D Patch cude pointer for GPU memory
  double* deviceP = NULL;
  cudaMalloc(&deviceP, m*n*w*w*sizeof(double));

  // Filtered image pointer for GPU memory
  double* deviceF = NULL;
  cudaMalloc(&deviceF, m*n*sizeof(double));

  FILE* fptr = fopen(argv[6], "r");
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i)
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j)
      fscanf(fptr, "%lf,", X+i*(n+w-1)+j);
      // X[i*(n+w-1)+j] = (i+j);
  fclose(fptr);

  // Add noise to input image
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
  char outputFileName[100] = "";
  sprintf(outputFileName, "../output_images/output_images_csv_txt/output_images_V1/%s_%d_%d_noisy.txt", argv[7], n, w);
  // printf("Writing noisy image to %s\n", outputFileName);
  printMatrixCsv(X, m+w-1, n+w-1, outputFileName);

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

  // Calculate residual image
  for(int i = 0; i < m; ++i) {
    for(int j = 0; j < n; ++j) {
      R[i*n+j] = F[i*n+j] - X[(i+(w-1)/2)*(n+w-1)+(j+(w-1)/2)];
    }
  }

  // Write filtered image to csv txt file
  // used by matlab script
  sprintf(outputFileName, "../output_images/output_images_csv_txt/output_images_V1/%s_%d_%d_denoised.txt", argv[7], n, w);
  // printf("Writing denoised image to %s\n", outputFileName);
  printMatrixCsv(F, m, n, outputFileName);

  sprintf(outputFileName, "../output_images/output_images_csv_txt/output_images_V1/%s_%d_%d_residual.txt", argv[7], n, w);
  // printf("Writing residual image to %s\n", outputFileName);
  printMatrixCsv(R, m, n, outputFileName);

  printf("%lf\n", milliseconds);

  // Deallocate CPU and GPU memory
  cudaFree(deviceP);
  cudaFree(deviceF);
  free(X);
  free(F);
  free(W);
  free(P);
  free(R);

  return 0;
}
