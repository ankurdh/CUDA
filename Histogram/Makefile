NVCC = /usr/local/cuda/bin/nvcc
CUDAPATH = /usr/local/cuda
NVCCFLAGS = -I$(CUDAPATH)/include -arch=sm_11
FLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm

VectorAdd:
	$(NVCC) $(NVCCFLAGS) $(LFLAGS) -o GPUHistogram.out GPUHistogram.cu
