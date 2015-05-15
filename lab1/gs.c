/*************************************************************************
*  Parallel Computing Assignment 1
*  Chris Jimenez
*  3/15/14
*
*  This program uses MPI to calcualte a set of unkowns(x_1...x_n)
*  by using the Jacobi method! That is, using initial values of
*  unknowns a set of new unkowns are calculated. The error
*  between the two sets is calculated and if it meets under
*  a given error threshold, then it is accepted! Otherwise,
*  the calcualtion keeps repeating until it is met by replacing
*  the old set of unkowns iwth the new set of unkowns
*
*  Functions used:
*      check_matrix()
*      get_input()
*      check_error()
*
*  MPI functions used:
*      MPI_Scatter
*      MPI_Init
*      MPI_Finalize
*      MPI_Allgatherv
*      MPI_Barrier
*
***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <assert.h>

/***** Globals ******/
float *a;       /* The coefficients */
float *x;       /* The unknowns */
float *b;       /* The constants */
float *diag_vec; /*diagonal vector of each row in a*/
float err;      /* The absolute relative error */
int num = 0;    /* number of unknowns */


/****** Function declarations */
void check_matrix();    /* Check whether the matrix will converge */
void get_input();       /* Read input from file */
int check_error();
/********************************/


/****************************************************************/
/* Calcualtes error between to sets of unkowns, or in this */
/* case X's */
int check_error( float *new_x, int num){
    int i, flag = 0;//set flag to 0, that is 'false'
    float abs_error;

    for(i = 0; i < num; i++){
        //calculate absolute error....
        abs_error = (new_x[i] - x[i])/new_x[i];

        if(abs_error > err){
            flag = 1;
            return flag;//immediately return true...
        }
    }
    return flag;//return false...
}

/*
 Conditions for convergence (diagonal dominance):
 1. diagonal element >= sum of all other elements of the row
 2. At least one diagonal element > sum of all other elements 
    of the row
 */
void check_matrix(){
    int bigger = 0;     /* Set to 1 if at least one diag element > sum  */
    int i, j;           //for for loops
    float sum = 0;      //sum of all elements besides aii
    float aii = 0;      //value at aii, elemts in diagonal matrix
    
    
    //for each diagonal in the matrix a...
    for(i = 0; i < num; i++){
        sum = 0;
        //aii is the absolutve value of a[i][i]
        aii = fabs(a[i*(num + 1)]);
        
        
        for(j = 0; j < num; j++){
            //not including the diagonal, add summ of elements at row
            if( j != i) sum += fabs(a[i*num + j]);
        }
        
        //check if matrix will not converge if  aii < sum
        if( aii < sum){
            printf("The matrix will not converge\n");
            exit(1);
        }
        
        
        //check if any element aii is bigger than the sum, in which case
        // bigger is incremented
        if(aii > sum) bigger++;
    }
    
    
    //if bigger is not >0, matrix will not converge...
    if( !bigger ){
        printf("The matrix will not converge\n");
        exit(1);
    }
}


/**
*   Read input from file
*/
void get_input(char filename[]){
    FILE * fp;
    int i ,j;  //for the for loops
    
    //attempt to open file...
    fp = fopen(filename, "r");
    
    //if file cant be open, do the following...
    if(!fp){
        printf("Cannot open file %s\n", filename);
        exit(1);
    }
    
    //number of unkowns/equations
    fscanf(fp,"%d ",&num);
    //relative error
    fscanf(fp,"%f ",&err);
    
    //////////////////////////////////////////////////////////
    /* Now, time to allocate the matrices and vectors */ /////
    //a
    //setup as contiguous array
    a = (float*)malloc(num * num * sizeof(float));
    
    //if a cannot be allocated, do the following...
    if(!a){
        printf("Cannot allocate a!\n");
        exit(1);
    }
    
    x = (float *) malloc(num * sizeof(float));
    
    //if x cannot be allocated, do the following...
    if(!x){
        printf("Cannot allocate x!\n");
        exit(1);
    }
    
    b = (float *) malloc(num * sizeof(float));
    
    //if b cannot be allocated, do the following...
    if( !b){
        printf("Cannot allocate b!\n");
        exit(1);
    }
    
    //allocate space or daig_vec
    diag_vec = (float *) malloc(num * sizeof(float));
    
    // The initial values of Xs
    
    for(i = 0; i < num; i++){
        fscanf(fp,"%f ", &x[i]);
    }
    
    //set up a as a contiguous linear array
    for(i = 0; i < num ; i++){
        for(j = 0; j < num; j++){
            fscanf(fp,"%f ",&a[i*num + j]);
            
            if(i == j){
                diag_vec[i] = a[i*num + j];
            }
        }
        
        /* reading the b element */
        fscanf(fp,"%f ",&b[i]);
    }
    
    //close the file....
    fclose(fp);
}


/**
 *  MAIN
 */
int main(int argc, char *argv[]){
    int nit = 0;    /* number of iterations */
    int i, j, k;   //for for loops

    
    if( argc != 2) {
        printf("Usage: gsref filename\n");
        exit(1);
    }
    /* Read the input file and fill the global data structure above */
    //argv[1] is the input file name...(small.txt, medium.txt, large.txt)
    get_input(argv[1]);
    
    /* Check for convergence condition */
    check_matrix();
    
    //Initialize MPI...
    int comm_sz;
    int my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    //MPI variables...
    int local_num_min = num/comm_sz;
    int num_extra = num % comm_sz;
    int sendcounts[comm_sz], displs[comm_sz], recv[comm_sz];
    int disp = 0;

    //create sendcounts[] and recv[] for MPI_Scatterv and MPI_Allgatherv
    for(i = 0; i < comm_sz; i++){
        if(i < num_extra){
            sendcounts[i] = local_num_min + 1;
        } else {
            sendcounts[i] = local_num_min;
        }

        recv[i] = sendcounts[i];

        displs[i] = disp;
        disp = disp + sendcounts[i];
    }
    

    //calculate local_num
    int local_num = (int)ceil((double)num/comm_sz); 

    //For each processes, allocate space for local(partial) x, b ,a
    float *local_x = (float *) malloc(local_num * sizeof(float));
    float *local_a = (float *) malloc(num * local_num * sizeof(float));
    float *local_b = (float *) malloc(local_num * sizeof(float));
    float *curr_x = (float *) malloc(num * sizeof(float));
    float *local_diag_vec = (float *) malloc(local_num * sizeof(float)); 

    //a is a contiguous array and we are scatter ROWs that is length num
    int num_elements_a = (local_num * num);
    MPI_Scatter(a, num_elements_a, MPI_FLOAT, 
            local_a, num_elements_a, MPI_FLOAT, 0, MPI_COMM_WORLD);

    //MPI process 0 scatter vector b  and matrix a to all processes
    MPI_Scatter(b, local_num, MPI_FLOAT, 
            local_b, local_num, MPI_FLOAT, 0, MPI_COMM_WORLD);


    //scatter diag vector...
    MPI_Scatter(diag_vec, local_num, MPI_FLOAT, 
            local_diag_vec, local_num, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    //MPI process 0 scatter current vector x to all processes
    MPI_Scatterv(x, sendcounts, displs, MPI_FLOAT, 
            local_x, recv[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    //initialize curr_x to x
    for(i = 0; i < num; i++){
        curr_x[i] = x[i];
    }

    ///////////////////////////////////////////////////////////////////
    //*Should be noted...*
    // sendcount[my_rank] is used in the loop becuase that is the number
    // of values given to each process, so we use that number to 
    // calcualte the new unkwowns...
    do{
        nit++;

        //set x to values of curr_x
        for(i = 0; i < num; i++){
            x[i] = curr_x[i];
        }


        //calculate local x's
        //each process calculate their local_x new values given currx
        for(i = 0; i < sendcounts[my_rank]; i++){
            int i_global = i; 
            for(k = 0; k < my_rank; k++){
                i_global += sendcounts[k];
            }

            //initially set to the corresponding constant
            local_x[i] = local_b[i]; 

            //calculate up to i_global...
            for(j = 0; j < i_global; j++){
                local_x[i] = local_x[i] - local_a[(i*num )+ j] * x[j];
      
            }

            //continue after i_global...
            for(j = i_global + 1; j < num; j++){
                local_x[i] = local_x[i] - local_a[(i*num )+ j] * x[j];
      
            }

            //divide local_x[i] by its corresponding diagonal element in a
            local_x[i] = local_x[i]/local_diag_vec[i];

        }

        //gather all local_x's from all processes and set to curr_x
        MPI_Allgatherv(local_x, sendcounts[my_rank], MPI_FLOAT, 
                curr_x, recv, displs, MPI_FLOAT, MPI_COMM_WORLD);


        //check error, if greater than threshold repeat...
    }while(check_error(curr_x, num));
    ///////////////////////////////////////////////////////////
    

    if( my_rank == 0){
        /* Writing to the stdout */
        /* Keep that same format */
        
        for(i = 0; i < num; i++){
            printf("%f\n", x[i]);
        }
        //print total number of iterations...
        printf("total number of iterations: %d\n", nit);

         free(x);
         free(a);
         free(b);
         free(diag_vec);
    }

     //clean up time...
     free(local_x);
     free(local_a);
     free(local_b);
     free(curr_x);
     free(local_diag_vec);
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize(); 
    return 0;
}