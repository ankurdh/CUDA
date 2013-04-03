#include<stdio.h>
#include<stdlib.h>

__device__ int gpuHistogram[10];

__global__ void computeGpuHistogram(int *arr, int noOfElements)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int noOfThreads = blockDim.x * gridDim.x;

	if(tid < 10)
		gpuHistogram[tid] = 0;

	__syncthreads();

	while(tid < noOfElements)
	{
		atomicAdd(gpuHistogram + arr[tid], 1);
		tid += noOfThreads;
	}
}


void checkError(cudaError_t error, char * function)
{
	if(error != cudaSuccess)
	{
	    printf("\"%s\" has a problem with error code %d and desc: %s\n", function, error, cudaGetErrorString(error));
	    exit(-1);
    }
} 

void readValue(int *value, char * msg, int lowerBound, int upperBound)
{
	while(1)
	{
		printf("%s(%d-%d): ", msg, lowerBound, upperBound);
		scanf("%d", value);
		
		if(*value <= upperBound && *value >= lowerBound)
		    return;	

		printf("Incorrect values. Enter again.\n");
	}
}

void fillArrayWithRandNos(int * arr, int noOfElements)
{
	int i;
	srand(5);												//for consistent numbers on every run.
	if(noOfElements < 20)
	{
		for(i = 0 ; i < noOfElements; ++i)
		{
            arr[i] = rand()%10;
			printf("%d   ", arr[i]);
		}
        printf("\n");
		return;
	}

	for(i = 0 ; i < noOfElements; ++i)
		arr[i] = rand()%10;
}

void computeHistogram(int *arr, int *histogram, int noOfElements)
{
	int i;
	for(i = 0 ; i < noOfElements ; ++i)
		++histogram[arr[i]];
}

bool cpuGpuResultsCompare(int *cpuResultsArray, int * histogramFromGPU)
{
	for(int i = 0 ; i < 10 ; i ++)
    	if(cpuResultsArray[i] != histogramFromGPU[i])
			return false;

	return true;
}

int main()
{
	int noOfElements = -1, i;
	int histogram[10] = {0,0,0,0,0,0,0,0,0,0}, histogramFromGPU[10];
	int *arr, *gpuArray;
    size_t size; 
	
	//have variables for threads per block, number of blocks.
    int threadsPerBlock = 0, blocksInGrid = 0;

    //create cuda event variables
    cudaEvent_t hostStart, hostStop, deviceStart, deviceStop;
    float timeDifferenceOnHost, timeDifferenceOnDevice;
    
    //create cuda events.
    cudaEventCreate(&hostStart);
	cudaEventCreate(&hostStop);
	cudaEventCreate(&deviceStart);
	cudaEventCreate(&deviceStop);

	printf("Enter the no. of elements to run test on: ");
	scanf("%d", &noOfElements);

    arr = (int *)malloc(noOfElements * sizeof(int));

	printf("Filling array with random numbers...\n");
	fillArrayWithRandNos(arr, noOfElements);

    printf("Computing histogram on CPU...\n");
	cudaEventRecord(hostStart, 0);
    computeHistogram(arr, histogram, noOfElements);
	cudaEventRecord(hostStop, 0);
    cudaEventElapsedTime(&timeDifferenceOnHost, hostStart, hostStop);

	//printf("Computation over. Results of CPU computation:\n");
	for(i = 0 ; i < 10 ; ++i)
        printf("No of %d: %d\n", i, histogram[i]);


	size = noOfElements * sizeof(int);
    checkError(cudaMalloc((void**)&gpuArray, size), "Mallocing array on GPU");

	checkError(cudaMemcpy(gpuArray, arr, size, cudaMemcpyHostToDevice), "Input array copy");
   
	while(1)
	{
		//create a proper grid block using dim3
	    readValue(&threadsPerBlock, "Enter no. of threads per block(input of 'P' will construct a P threaded linear block)", 4, 256);
		readValue(&blocksInGrid, "Enter no. of blocks in grid(input of 'P' will construct linear grid with P blocks)", 0, 65535/threadsPerBlock+1);

		cudaEventRecord(deviceStart, 0);
	 	computeGpuHistogram<<<blocksInGrid, threadsPerBlock>>>(gpuArray, noOfElements);   
	    cudaThreadSynchronize();
	    cudaEventRecord(deviceStop, 0);
		cudaEventElapsedTime(&timeDifferenceOnDevice, deviceStart, deviceStop);
	
		cudaMemcpyFromSymbol(&histogramFromGPU,"gpuHistogram", sizeof(histogramFromGPU), 0, cudaMemcpyDeviceToHost);
	
	  	if(cpuGpuResultsCompare(histogram, histogramFromGPU))
			printf("GPU and CPU results match\n");
		else
			printf("GPU and CPU results don't match\n");
	
		printf("CPU & GPU stats: \n");
		for(i = 0 ; i < 10 ; ++i)
	        printf("No of %ds: %d %d\n", i, histogram[i], histogramFromGPU[i]);
		
		printf("Time on CPU : %5.5f, Time on GPU: %5.5f\n", timeDifferenceOnHost, timeDifferenceOnDevice);

		printf("\n-----------------------------------------------\n");
		printf("Speedup: %5.5f\n", timeDifferenceOnHost/timeDifferenceOnDevice);
		printf("-----------------------------------------------\n");
	
		char c = 'n';
	 	printf("Again?(y/n): ");
		while(true)
		{
		   c = getchar();
	       if(c == 'y' || c == 'n')
	   	  	  break;
	    }
	   
	    if(c == 'n')
	       break;
	}	 
	printf("\n");
    free(arr);
	cudaFree(gpuArray);

    return 0;
}
