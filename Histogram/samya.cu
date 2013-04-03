#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
//#define N 100
__device__ int gpu_hist[10];
__global__ void gpuhistogram(int *a,int N)
{
	int *ptr;
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	int numthr=blockDim.x*gridDim.x;

	if(tid==0)
			for(int i=0;i<10;i++)
						gpu_hist[i]=0;
	
	__syncthreads();
						
	while(tid<N)
	{
		ptr=&gpu_hist[a[tid]];
		atomicAdd(ptr,1);
		tid+=numthr;
	}
}
int main()
{
	int B,T;
	int *a;
	int *deva;
	int N;
	int hist[10],cist[10];
	for(int i=0;i<10;i++)
	{cist[i]=0;hist[i]=0;}
	printf("Enter the number of elements .\n");
	scanf("%d",&N);
	printf("Enter the number of Blocks and Threads .\n");
	again:;
	printf("Blocks:");
	scanf("%d",&B);
	printf("Threads:\n");
	scanf("%d",&T);
	if(B*T<N)
	{printf("The number of blocks and threads is less please enter again.\n");
	goto again;
	}
	cudaEvent_t start,stop;
	float cput,gput;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int size=N*sizeof(int);
	a=(int*)malloc(size);
	srand(1);
	for(int i=0;i<N;i++)
		a[i]=rand()%10;
		cudaMalloc((void**)&deva,size);
		cudaMemcpy(deva,a,size,cudaMemcpyHostToDevice);
		cudaEventRecord(start,0);
		gpuhistogram<<<B,T>>>(deva,N);
		//cudaThreadSynchronize();
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gput,start,stop);
		cudaMemcpyFromSymbol(&hist,"gpu_hist",sizeof(hist),0,cudaMemcpyDeviceToHost);
		printf("GPU execution completed.\n");
		int l;
		for (int i=0;i<10;i++)
		{
			cist[i]=0;
		}
		cudaEventRecord(start,0);
		for(int i=0;i<N;i++)
		{
			l=a[i];
			cist[l]++;
		}
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&cput,start,stop);

		for(int i=0;i<10;i++)
				{printf("Number of %d's = gpu: %d   cpu: %d \n",i,hist[i],cist[i]);
					}
					free(a);
					cudaFree(deva);
					printf("CPUtime= %f and GPUtime= %f.\n",cput,gput);
					return 0;

					/////////
}

