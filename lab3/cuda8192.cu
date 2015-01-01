 /////////////////////////////////////////////////////////////////////////
//  Parallel Computing Assignment 3
//  Chris Jimenez
//  5/1/14
//  This CUDA program finds the max integer in an array of random integers.
// 	This program DOES NOT use shared meemory and DOES NOT take thread
// 	divergaence in to consideration.
//
/////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//define numebr of integers...
#define NUM_OF_INTEGERS 8192
//define max integer
#define MAX 100000

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
__global__ void get_max(int *array);
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
__global__ void get_max(int *array){
	int temp;
	int index = threadIdx.x + (blockDim.x * blockIdx.x);
	int nTotalThreads = NUM_OF_INTEGERS;	

	while(nTotalThreads > 1){
		int halfPoint = nTotalThreads / 2;	// divide by two
		// only the first half of the threads will be active.
		if (index < halfPoint){
			temp = array[ index + halfPoint ];
			if (temp > array[ index ]) {
				array[index] = temp;
			}
		}
		__syncthreads();


		nTotalThreads = nTotalThreads / 2;	// divide by two.
	}
}

/*******************************************************/
//Main function.....
int main(int argc, char *argv[]){
	int *h_array;	//array of random integers....
	int *d_array;	//device copy...

	printf("Initializing data...\n");
	//allocating space for the array on host
	h_array = (int *) malloc(NUM_OF_INTEGERS * sizeof(int));

	//fill in random array
	fill_array(h_array);

	//allocate space for array and resultmax on device
	cudaMalloc( (void **)&d_array, sizeof(int) * NUM_OF_INTEGERS );

	//Copy array from host to device...
	cudaMemcpy(d_array, h_array, sizeof(int) * NUM_OF_INTEGERS, cudaMemcpyHostToDevice);

	//call kernel! using for loop
	get_max<<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(d_array);
	
	//Copy array from device to host...
	cudaMemcpy(h_array, d_array, sizeof(int) * NUM_OF_INTEGERS, cudaMemcpyDeviceToHost);

	//print max value...
	printf("The max integer in the array is: %d\n", h_array[0]);

	printf("Cleaning up...\n");
	free(h_array);
	cudaFree(d_array);

	return 0;
}