#include<stdio.h>
#include<stdlib.h>

__global__ void matTranspose(int *matrixA, int *matrixB, int matSize)
{
    int threadCol = blockIdx.x * blockDim.x + threadIdx.x;
    int threadRow = blockIdx.y * blockDim.y + threadIdx.y;
    
    int srcIndex = threadRow*matSize+threadCol;
	int dstnIndex = threadCol*matSize+threadRow;

	matrixB[dstnIndex] = matrixA[srcIndex];

}

void printMatrix(int *matrix, int size, char * matrixName)
{

    if(size > 10)
      return;

    int i = 0;
    printf("Printing Matrix: %s\n", matrixName);
    for( ; i < size * size ; i ++)
    {
        if(i % size == 0)
            printf("\n");

        printf("%-3d  ", matrix[i]);
    }

    printf("\n\n");
}

void checkError(cudaError_t error, char * function)
{

        if(error != cudaSuccess)
        {
                printf("\"%s\" has a problem with error code %d and desc: %s\n", function, error, cudaGetErrorString(error));
                exit(-1);
        }
}

bool checkIfMatricesEqual(int * mat1, int * mat2, int matSize)
{
    int i = 0;
    for( ; i < matSize; i++)
       if(mat1[i] != mat2[i]){
           printf("values different for i: %d\n", i);
		   printf("mat1[i] = %d, mat2[i] = %d\n", mat1[i], mat2[i]);		   
		   return false;
	   }

    return true;
}

void readValue(int *value, char * msg, int lowerBound, int upperBound)
{
    while(true)
    {
        printf("%s(%d-%d): ", msg, lowerBound, upperBound);
        scanf("%d", value);
        
        if(*value <= upperBound && *value >= lowerBound)
            return;
    }        
}

int main()
{

   //have variables for threads per block, number of blocks.
   int threadsPerBlock = 0, blocksInGrid = 0;

   //create cuda event variables
   cudaEvent_t hostStart, hostStop, deviceStart, deviceStop;
   float timeDifferenceOnHost, timeDifferenceOnDevice;

   //program variables
   int matrixSize = 0;
   size_t size;                     //variable to have the size of arrays on device
   int *matA, *matB, *matC, *matCFromGPU;   //matrices for host
   int *gpuMatA, *gpuMatB, *gpuMatC;            //matrices for Device

   //initialize cuda timing variables
   cudaEventCreate(&hostStart);
   cudaEventCreate(&hostStop);
   cudaEventCreate(&deviceStart);
   cudaEventCreate(&deviceStop);
  
   printf("Enter the size of the matrix: ");
   scanf("%d", &matrixSize);

   //calculate the size required on GPU
   size = matrixSize * matrixSize * sizeof(int);

   matA = (int *)malloc(matrixSize * sizeof(int) * matrixSize);
   matB = (int *)malloc(matrixSize * sizeof(int) * matrixSize);
   matC = (int *)malloc(matrixSize * sizeof(int) * matrixSize);

   for(int i = 0 ; i < matrixSize * matrixSize; i ++)
         	  matA[i] = (i*2)%10;

   printMatrix(matA, matrixSize, "Matrix A");
   //printMatrix(matB, matrixSize, "Matrix B");
   printf("Transposing matrix on CPU...\n");
   cudaEventRecord(hostStart, 0);
   //matrix multiplication code goes here.
   for(int i = 0 ; i < matrixSize ; i ++)
   {
	   for(int j = 0 ; j < matrixSize ; j ++)
	   {
    	   //printf("Src Indx: %d, Dstn Indx: %d\n", i*matrixSize+j, j*matrixSize+i);
           matB[j*matrixSize+i] = matA[i*matrixSize+j];

	   }
   }
   cudaEventRecord(hostStop, 0);
   cudaEventElapsedTime(&timeDifferenceOnHost, hostStart, hostStop);
   printf("Matrix transpose over. Time taken on CPU: %5.5f\n", timeDifferenceOnHost);
   
   printMatrix(matB, matrixSize, "Transpose Matrix");
   
   //allocate memory on GPU
   checkError(cudaMalloc((void**)&gpuMatA, size), "Malloc for Matrix A");
   checkError(cudaMalloc((void**)&gpuMatB, size), "Malloc for Matrix B");
   //checkError(cudaMalloc((void**)&gpuMatC, size), "Malloc for Matrix C");

   //copy the matrix A and matrix B
   checkError(cudaMemcpy(gpuMatA, matA, size, cudaMemcpyHostToDevice), "Matrix A Copy");
   //checkError(cudaMemcpy(gpuMatB, matB, size, cudaMemcpyHostToDevice), "Matrix B Copy");

   bool done = false;
 
   while(!done)   
   {
	   matCFromGPU = (int *)malloc(matrixSize * sizeof(int) * matrixSize);

	   //create a proper grid block using dim3
	   readValue(&threadsPerBlock, "Enter no. of threads per block(input of 'P' will construct PxP threads in block)", 4, 32);
	   readValue(&blocksInGrid, "Enter no. of blocks in grid(input of 'P' will construct PxP blocks)", (matrixSize + threadsPerBlock -1)/threadsPerBlock, 65535);
	   printf("Threads Per block: %d, Blocks in grid: %d\n", threadsPerBlock, blocksInGrid); 
	   printf("Multiplying matrices on GPU..\n");
	   dim3 blocks(threadsPerBlock, threadsPerBlock);                                                   
	   dim3 grid(blocksInGrid, blocksInGrid); //(matrixSize + threadsPerBlock - 1/blocks.x), (matrixSize + blocks.y - 1/blocks.y));
	
	   //call the kernels to execute
	   cudaEventRecord(deviceStart, 0);
	   printf("Total linear threads: %d\n", blocksInGrid*threadsPerBlock);
	   matTranspose<<<grid, blocks>>>(gpuMatA, gpuMatB, matrixSize);
	   cudaEventRecord(deviceStop, 0);
	   cudaEventSynchronize(deviceStop);
	
	   cudaEventElapsedTime(&timeDifferenceOnDevice, deviceStart, deviceStop);
	
	   //copy the result back into host memory
	   checkError(cudaMemcpy(matCFromGPU, gpuMatB, size, cudaMemcpyDeviceToHost), "Matrix C Copy from device to Host");
	
	   if(checkIfMatricesEqual(matB, matCFromGPU, matrixSize))
	      printf("Kernels correct!\n");
	   else
	      printf("Kernel logic wrong!\n");
	
	   printf("Finished transposing on GPU. Time taken: %5.5f\n", timeDifferenceOnDevice);   
	   printf("Speedup: %5.5f\n", (float)timeDifferenceOnHost/timeDifferenceOnDevice);
	
	   printMatrix(matCFromGPU, matrixSize, "Transpose Matrix from GPU");
	
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
	
	   free(matCFromGPU);
   }
     
   free(matA);
   free(matB);
   free(matC);
	
   cudaEventDestroy(deviceStart);
   cudaEventDestroy(deviceStop);
   cudaEventDestroy(hostStart);
   cudaEventDestroy(hostStop);
 
   return 0;
   
}
