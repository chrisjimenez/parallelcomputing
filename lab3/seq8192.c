 /////////////////////////////////////////////////////////////////////////
//  Parallel Computing Assignment 3
//  Chris Jimenez
//  5/1/14
//
/////////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include <stdlib.h>
#include <time.h>


//define numebr of integers...
#define NUM_OF_INTEGERS 8192
#define MAX 100000


/****** Function declarations */
void fill_array();
int get_random_num();
/********************************/

/*******************************************************/
/* Function fills the givne array a with random integers */
void fill_array(int *a){
	int i;
	time_t t;
    
   	/* Intializes random number generator */
   	srand((unsigned) time(&t));
    
	for(i = 0; i < NUM_OF_INTEGERS; i++){
		a[i] = random() % MAX;
	}
}


/*******************************************************/
int main(int argc, char *argv[]){
	int i;
	int *rand_array;	//array of random integers....

	int max_num = 0;

	//allocating space for the random array
	rand_array = (int *) malloc(NUM_OF_INTEGERS * sizeof(int));

	//fill in random array
	fill_array(rand_array);

	for( i = 0; i < NUM_OF_INTEGERS; i++){
		if(rand_array[i] > max_num){
			max_num = rand_array[i];
		}
	}

	printf("%d\n", max_num);

	return 0;
}