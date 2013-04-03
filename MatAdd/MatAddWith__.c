#include<stdio.h>
#include<stdlib.h>

void printMatrix(int **matrix, int size, char * matrixName)
{
    int i = 0, j;
    printf("Printing Matrix: %s\n", matrixName);
    for( ; i < size ; i ++)
    {
        for(j = 0 ; j < size ; j ++)
        {
            printf("%d  ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

int main()
{
   
   //Have some variables required for loop counters.
   int i, j;
   
   //program variables
   int matrixSize = 0;
   size_t deviceMatrixSize;			//variable to have the size of arrays on device
   int **matA, **matB, **matC, **matCFromGPU;	//matrices for host
   int *gpuMatA, *gpuMatB, *gpuMatC;		//matrices for Device 
   
   printf("Enter the size of the matrix: ");
   scanf("%d", &matrixSize);
   
   matA = (int **)malloc(matrixSize * sizeof(int));
   matB = (int **)malloc(matrixSize * sizeof(int));
   matC = (int **)malloc(matrixSize * sizeof(int));

   for(i = 0 ; i < matrixSize ; i ++)
   {
      matA[i] = (int *)malloc(matrixSize * sizeof(int));
      matC[i] = (int *)malloc(matrixSize * sizeof(int));
      matB[i] = (int *)malloc(matrixSize * sizeof(int));
   }
   
   for(i = 0 ; i < matrixSize ; i ++)
     for(j = 0 ; j < matrixSize; j ++)
         matA[i][j] = matB[i][j] = i*2 + j;
         
   printMatrix(matA, matrixSize, "Matrix-A");
   printMatrix(matB, matrixSize, "Matrix-B");
         
   for(i = 0 ; i < matrixSize ; i ++)
       for(j = 0 ; j < matrixSize ; j ++)
           matC[i][j] = matA[i][j] + matB[i][j];
   
   printMatrix(matC, matrixSize, "Summation Matrix");
   
   for(i = 0 ; i < matrixSize ; i ++)
   {   
      free(matA[i]);
      free(matC[i]);
      free(matB[i]);
   }   

   free(matA);
   free(matB);
   free(matC);

   return 0;
}

