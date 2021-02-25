#include "utilities.h"

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

void printMatrixCsv(float* A, int m, int n, char* name) {
  FILE* ptr = fopen(name, "w");
  if(ptr == NULL) {
    printf("Failed to create output file. Exiting!\n");
    return;
  }
  for(int i = 0; i < m; ++i) {
    fprintf(ptr, "%f", A[i*n]);
    for(int j = 0; j < n; ++j) {
      fprintf(ptr, ",%f", A[i*n+j]);
    }
    fprintf(ptr,"\n");
  }
  fclose(ptr);
}
