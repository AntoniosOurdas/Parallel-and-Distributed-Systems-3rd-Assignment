#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

void printMatrixFloat(float* A, int n, int m) {
  for(int j = 0; j < m*11; ++j) {
    printf("-");
  }
  printf("\n");
  for(int i = 0; i < n; ++i) {
    printf("| ");
    for(int j = 0; j < m; ++j) {
      if(A[i*m+j] < 0.0)
        printf("%f ", A[i*m+j]);
      else
        printf(" %f ", A[i*m+j]);
		}
		printf(" |\n");
	}
  for(int j = 0; j < m*11; ++j) {
    printf("-");
  }
  printf("\n");
}

void printMatrixMatlab(float* A, int m, int n, char* name) {
  printf("%s = [", name);
  for(int i = 0; i < m; ++i) {
    for(int j = 0; j < n; ++j)
      printf("%f ", A[i*n+j]);
    printf(";");
  }
  printf("];\n");
}

float gaussianRand(double sigma) {
  float sum = 0.0;
  for(int i = 0; i < 10; ++i) {
    sum += -2.0*sqrt(12)*sigma*(float)rand()/RAND_MAX+1.0*sqrt(12)*sigma;
  }
  sum /= 10.0;
  return sum;
}

int main(int argc, char* argv[]) {

  // Get image size
  int m = atoi(argv[1]);
  int n = atoi(argv[2]);

  // Get patch size
  int w = atoi(argv[3]);

  // Image 1D array (row major) (extended to fit patches)
  float* X = (float*)malloc((m+w-1)*(n+w-1)*sizeof(float));

  char* filename = "";
  switch(m) {
    case 64:
      filename = "lena_64.txt";
      break;
    case 128:
      filename = "lena_128.txt";
      break;
    case 256:
      filename = "lena_256.txt";
      break;
    case 512:
      filename = "lena_512.txt";
      break;
  }
  // printf("Reading: %s\n", filename);
  // Read input image from txt file of doubles
  FILE* fptr = fopen(filename, "r");

  for(int i = (w-1)/2; i < m+(w-1)/2; ++i)
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j)
      fscanf(fptr, "%f,", X+i*(n+w-1)+j);
      // X[i*(n+w-1)+j] = (i % (m / 8) == 0 || j % (n / 8) == 0) ? 1.0 : 0.0;
  fclose(fptr);

  printf("figure;\n");
  // Print original image in matlab command format
  printMatrixMatlab(X, m+w-1, n+w-1,"X");

  // Add gaussian noise to image (with standard deviation 0.04)
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i)
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j)
      X[i*(n+w-1)+j] += gaussianRand(0.04);


  // Fill edges mirroring the inside of image

  // Right and left part
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i) {

    for(int j = 0; j < (w-1)/2; ++j) {
      X[i*(n+w-1)+j] = X[i*(n+w-1)+(w-j-1)];
    }

    for(int j = 1; j <= (w-1)/2 ; ++j) {
      X[i*(n+w-1)+((n+w-1)-j)] = X[i*(n+w-1)+((n+w-1)-(w-(j-1)))];
    }

  }

  // Upper and lower part
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i) {

    for(int j = 0; j < (w-1)/2; ++j) {
      X[j*(n+w-1)+i] = X[(w-j-1)*(n+w-1)+i];
    }

    for(int j = 1; j <= (w-1)/2 ; ++j) {
      X[((n+w-1)-j)*(n+w-1)+i] = X[((n+w-1)-(w-(j-1)))*(n+w-1)+i];
    }

  }

  // Corners ((w-1)/2 by (w-1)/2)


  // Print noisy image in matlab command format
  // printMatrixDouble(X, m+w-1, n+w-1);
  printMatrixMatlab(X, m+w-1, n+w-1, "Xn");

  // Define denoised image
  float* F = (float*)malloc(m*n*sizeof(float));

  // Get gaussian patch standard deviation
  float patchSigma = 1.0;
  sscanf(argv[4],"%f",&patchSigma);


  // Define and fill normalized gaussian patch window
  float* W = (float*)malloc(w*w*sizeof(float));
  float sum = 0.0;
  for(int i = 0; i < w; ++i) {
    for(int j = 0; j < w; ++j) {
      W[i*w+j] = exp((-pow((float)(i-w/2)/(float)w, 2)-pow((float)(j-w/2)/(float)w, 2))/(2.0*patchSigma));
      // W[i*w+j] = 1.0;
      sum += W[i*w+j];
    }
  }

  for(int i = 0; i < w; ++i)
    for(int j = 0; j < w; ++j)
      W[i*w+j] /= sum;

  // Print patch window in matlab command format
  printMatrixMatlab(W, w, w, "w");
  // printMatrixDouble(W, w, w);


  // Weight w(x,y)
  float Wxy = 0.0;
  // Sum of weights Z(i)
  float Zx = 0.0;
  // Distance of patches
  float D = 0.0;
  // Get standard deviation of gaussian filter
  // (G(a) as mentioned in excercise formula)
  float filtSigma = 1.0;
  sscanf(argv[5],"%f",&filtSigma);

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
                // printf("Distance: %lf\n", D);
              }
            }
            // printf("%lf\n", D);
            // Set weight which shows similarity
             // of patches (i,j) and (k,l)
            Wxy = exp(-D/(filtSigma*filtSigma));
            // Add weight to Z(i) for normalization
            Zx += Wxy;
            // Add weighted pixel to final sum
            F[(i-(w-1)/2)*n+(j-(w-1)/2)] += Wxy * X[k*(n+w-1)+l];
            // printf("Wxy = %f\n", Wxy);
        }
      }
      // printf("Done: %lf\n", Zx);
      // Normalize sum
      // printf("%lf\n", F[i*n+j]);
      // Normalize by dividing with Z(i)
      // This is our final denoised image pixel value
      // After this we continue to the next pixel
      F[(i-(w-1)/2)*n+(j-(w-1)/2)] /= Zx;
      // printf("Done2: %lf\n", F[i*n+j]);
    }
  }
  // printMatrixDouble(F, m, n);
  printMatrixMatlab(F, m, n, "F");

  printf("subplot(1,3,1);imshow(X,[]);title('Original');subplot(1,3,2);imshow(Xn,[]);title('Noisy');subplot(1,3,3);imshow(F,[]);title('Denoised');\n");
  printf("\n%% V0 run time: %f\n", calculateExecutionTime(start_time));
  return 0;
}
