#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

void printMatrixInt(int* A, int n, int m);
void printMatrixDouble(double* A, int n, int m);
void printMatrixMatlab(double* A, int m, int n, char* name);
double gaussianRand(double sigma);
void printMatrixCsv(double* A, int m, int n, char* name);
