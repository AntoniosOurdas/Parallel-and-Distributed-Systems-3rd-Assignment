#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

int main(int argc, char* argv[]) {

  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  // Patch size
  int w = atoi(argv[3]);
  double low = -5.0, high = 5.0;
  // Image array (extended to fit patches)
  double* X = (double*)malloc((m+w-1)*(n+w-1)*sizeof(double));

  for(int i = (w-1)/2; i < m+(w-1)/2; ++i)
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j)
      X[i*(n+w-1)+j] = (double)(exp(-pow((double)(i-m/2)/(double)m,2.0)-pow((double)(j-n/2)/(double)n,2.0)));


  double* F = (double*)malloc(m*n*sizeof(double));

  // Patch window
  double* W = (double*)malloc(w*w*sizeof(double));
  for(int i = 0; i < w; ++i)
    for(int j = 0; j < w; ++j)
      W[i*w+j] = exp(-pow((double)(i-w/2)/(double)w, 2)-pow((double)(j-w/2)/(double)w, 2));

  printMatrixMatlab(W, w, w, "w");


  // Weight w(x,y)
  double Wxy = 0.0;
  double Zx = 0.0;
  double D = 0.0;
  double sigma = 1.0;
  // Each pixel (i,j) will be the weighted sum of all other pixels
  for(int i = (w-1)/2; i < m+(w-1)/2; ++i) {
    for(int j = (w-1)/2; j < n+(w-1)/2; ++j) {
      // Set value of pixel to 0
      // so as to start the summation
      // F[i*n+j] = 0.0;
      // Then iterate all other pixels
      // and calulcate the weight Wxy
      // Zx = 0.0;
      printf("(%d, %d):\n", i, j);
      for(int k = (w-1)/2; k < m+(w-1)/2; ++k) {
        for(int l = (w-1)/2; l < n+(w-1)/2; ++l) {
          printf("\t(%d, %d):\n", k, l);
          // if((i*n+j) != (k*n+l)) {
            Wxy = 0.0;
            // Iterate all patch elements
            D = 0.0;
            for(int p = -(w-1)/2; p <= (w-1)/2; ++p) {
              for(int q = -(w-1)/2; q <= (w-1)/2; ++q) {
                printf("\t\t(%d, %d), (%d, %d)\n", i+p, j+q, k+p, l+q);
                D += W[(p+(w-1)/2)*w+(q+(w-1)/2)]*pow((X[(i+p)*(n+w-1)+(j+q)] - X[(k+p)*(n+w-1)+(l+q)]),2);
              }
            }
            Wxy = exp(-D/(sigma*sigma));
            Zx += Wxy;
            F[i*n+j] += Wxy * X[k*n+l];
          // }
        }
      }
      // Normalize sum
      F[i*(n+w-1)+j] /= Zx;
    }
  }

  printMatrixDouble(F, m, n);
  return 0;
}
