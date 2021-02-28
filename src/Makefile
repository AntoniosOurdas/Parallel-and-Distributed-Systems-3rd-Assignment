CC=gcc
NVCC=nvcc
LIBRARIES = -lm

V0:
	$(CC) V0.c utilities.c -o V0 $(LIBRARIES)

V1:
	$(NVCC) V1.cu utilities.cu -o V1 $(LIBRARIES)

V2:
	$(NVCC) V2.cu utilities.cu -o V2 $(LIBRARIES)

all: V0 V1 V2

.PHONY: clean

clean:
	rm -f V0 V1 V2
