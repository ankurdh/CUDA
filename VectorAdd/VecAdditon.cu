#include<stdio.h>
#include<cuda.h>

#define N 10

__global__ void vecAdd(int *a, int *b, int *c)
{
    int id = blockIdx.x;
    if(id < N)
	c[id] = a[id] + b[id];
}

void checkError(cudaError_t error, char * function)
{

	if(error != cudaSuccess)
	{
		printf("\"%s\" has a problem with error code %d and desc: %s\n", function, error, cudaGetErrorString(error));
		exit(-1);
	}
}

int main()
{
	int a[N], b[N], c[N];
	int *deviceA, *deviceB, *deviceC;
	int i; //a variable for use in loops.
	size_t size = N * sizeof(int);

	//put some value in the 'a' & 'b' arrays
	for(i = 0 ; i < N ; i ++)
	{
		a[i] = i;
		b[i] = i;
	}

	//initialize the memory on GPU
	checkError(cudaMalloc((void**)&deviceA, size), "Cuda Malloc for deviceA");
	checkError(cudaMalloc((void**)&deviceB, size), "Cuda Malloc for deviceB");
	checkError(cudaMalloc((void**)&deviceC, size), "Cuda Malloc for deviceC");

	checkError(cudaMemcpy(deviceA, a, size, cudaMemcpyHostToDevice), "Cuda MemCpy for DeviceA");
	checkError(cudaMemcpy(deviceB, b, size, cudaMemcpyHostToDevice), "Cuda MemCpy for DeviceB");	

	vecAdd<<<N , 1>>>(deviceA, deviceB, deviceC);

	checkError(cudaMemcpy(c, deviceC, size, cudaMemcpyDeviceToHost), "Cuda MemCpy for DeviceC");

	for(i = 0 ; i < N ; i ++)
		printf("c[%d] = %d\n", i , c[i]);

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);	

	return 0;
}
