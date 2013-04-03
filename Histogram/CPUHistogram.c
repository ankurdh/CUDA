#include<stdio.h>
#include<stdlib.h>

/*void checkError(cudaError_t error, char * function)
{
	if(error != cudaSuccess)
	{
	    printf("\"%s\" has a problem with error code %d and desc: %s\n", function, error, cudaGetErrorString(error));
	    exit(-1);
    }
}*/ 

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

int main()
{
	int noOfElements = -1, i;
	int histogram[10] = {0,0,0,0,0,0,0,0,0,0};
	int *arr;

	printf("Enter the no. of elements to run test on: ");
	scanf("%d", &noOfElements);

    arr = (int *)malloc(noOfElements * sizeof(int));

	fillArrayWithRandNos(arr, noOfElements);

    computeHistogram(arr, histogram, noOfElements);

    printf("\n");

	for(i = 0 ; i < 10 ; ++i)
        printf("No of %d: %d\n", i, histogram[i]);

    return 0;
}
