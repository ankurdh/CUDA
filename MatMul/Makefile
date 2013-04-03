NVCC = /usr/local/cuda/bin/nvcc
CUDAPATH = /usr/local/cuda
NVCCFLAGS = -I$(CUDAPATH)/include
FLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm

VectorAdd:
	$(NVCC) $(NVCCFLAGS) $(LFLAGS) -o MatMul.out MatMul.cu
