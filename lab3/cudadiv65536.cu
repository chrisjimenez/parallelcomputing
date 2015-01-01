 /////////////////////////////////////////////////////////////////////////
//  Parallel Computing Assignment 3
//  Chris Jimenez
//  5/1/14
//  This CUDA program finds the max integer in an array of random integers.
// 	This program DOES NOT use shared meemory and DOES take thread
// 	divergaence in to consideration. The modification can be seen
//  in the kernel function with the use of the WARP_SIZE defined var.
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
	int nTotalThreads = NUM_OF_INTEGERS;	// Total number of active threads

	while(nTotalThreads > WARP_SIZE)
	{
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

	//	at this point...nTotalThreads == 32
	// 	that means that array[0:31] has the top 
	// 	32 values...
}

/*******************************************************/
//Main function.....
int main(int argc, char *argv[]){
	int *h_array;	//array of random integers....
	int *d_array;	//device copy...
	int max = 0;

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

	//given the top 32 largest numbers, search through to get max...
	for(int i = 0; i < WARP_SIZE; i++){
		if( max < h_array[i]){
			max = h_array[i];
		}

	}	

	//print max value...
	printf("The max integer in the array is: %d\n", h_array[0]);

	printf("Cleaning up...\n");
	free(h_array);
	cudaFree(d_array);

	return 0;
}