 /////////////////////////////////////////////////////////////////////////
//  Parallel Computing Assignment 3
//  Chris Jimenez
//  5/1/14
//  This CUDA program finds the max integer in an array of random integers.
// 	This program DOES use shared meemory and DOES take thread
// 	divergaence in to consideration.
//
/////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


//define numebr of integers...
#define NUM_OF_INTEGERS 65536
//define max integer
#define MAX 100000
#define WARP_SIZE 32

///////////////////////////////////
/*The folllowing is dependent on whatever GPU this program is running on
  if runnign on the NYU GPU's, the max threads per block is 512.
  RUnning on a NVIDIA GeForce GT 650M(on personal machine), the max threads
  per block is 1024 
*/
#define THREADS_PER_BLOCK 512
#define NUM_BLOCKS NUM_OF_INTEGERS/THREADS_PER_BLOCK


/****** Function declarations */
void fill_array();
__global__ void get_max(int *array, int *max_results);
/********************************/
/////////////////////////////////////////////////////////

/*******************************************************/
/* Function fills the givne array a with random integers */
void fill_array(int *a){
	int i;
	time_t t;
   
   	/* Intializes random number generator */
   	srand((unsigned) time(&t));

	for(i = 0; i < NUM_OF_INTEGERS; i++){
		a[i] = random() % MAX;;
	}
}

/*******************************************************/
/* Kernel Function finds the max integer in given array by 
	using reduction technique. Ultimately, the largest 
	will be located at the 0th position of the array */
__global__ void get_max(int *array, int *max_results){
	int  temp;
	__shared__ int  max[THREADS_PER_BLOCK];
	int index = threadIdx.x + (blockDim.x * blockIdx.x);
	max[threadIdx.x] = array[index];
	__syncthreads();

	int nTotalThreads = blockDim.x;	// Total number of active threads

	while(nTotalThreads > WARP_SIZE)
	{
		int halfPoint = nTotalThreads/2;	// divide by two
		// only the first half of the threads will be active.
		if (threadIdx.x < halfPoint)
		{
			temp = max[threadIdx.x + halfPoint];
			if (temp > max[threadIdx.x]) {
				max[threadIdx.x] = temp;
			}

		}
		__syncthreads();

		nTotalThreads = nTotalThreads/2;	// divide by two.
	}

		max_results[blockIdx.x] = max[0];
}



/*******************************************************/
int main(int argc, char *argv[]){
	int *h_array, *h_resultmax;	//array of random integers....
	int *d_array, *d_resultmax;
	int max;
	int i;

	printf("Initializing data...\n");
	//allocating space for the array on host
	h_array = (int *) malloc(NUM_OF_INTEGERS * sizeof(int));
	h_resultmax = (int *)malloc(sizeof(int) * NUM_BLOCKS);

	//fill in random array
	fill_array(h_array);

	//allocate space for array and resultmax on device
	cudaMalloc( (void **)&d_array, sizeof(int) * NUM_OF_INTEGERS );
	cudaMalloc( (void **)&d_resultmax, sizeof(int) * NUM_BLOCKS );

	//Copy array from host to device...
	cudaMemcpy(d_array, h_array, sizeof(int) * NUM_OF_INTEGERS, cudaMemcpyHostToDevice);

	//call kernel! using for loop
	get_max<<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(d_array, d_resultmax);
	
	//Copy array from host to device...
	cudaMemcpy(h_resultmax, d_resultmax, sizeof(int)*NUM_BLOCKS, cudaMemcpyDeviceToHost);

	//Given the max from each threadblock, find max!
	max = h_resultmax[0];
	for (i = 1 ; i < NUM_BLOCKS; i++){
		if (h_resultmax[i] > max) max = h_resultmax[i];
	}


	//print max value...
	printf("The max integer in the array is: %d\n", max);

	printf("Cleaning up...\n");
	free(h_array);
	free(h_resultmax);

	cudaFree(d_array);
	cudaFree(d_resultmax);

	return 0;
}