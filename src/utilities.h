#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

void printMatrixInt(int* A, int n, int m);
void printMatrixFloat(float* A, int n, int m);
void printMatrixMatlab(float* A, int m, int n, char* name);
float gaussianRand(double sigma);
void printMatrixCsv(float* A, int m, int n, char* name);
