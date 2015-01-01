 /////////////////////////////////////////////////////////////////////////
//  Parallel Computing Assignment 2
//  Chris Jimenez
//  4/10/14
//
//  This program uses OpenMP to solve a modified version of the traveling 
//  salesman problem, where in this case the salesman does not need to return
//  to their starting city.
//  
//  Functions used:
//      get_input()
//      init_path();
//      init_best_path(Path *path);
//      add_city()
//      print_path()
//      check_best_path()
//      remove_last()
//      copy_paths()
//      compute_path( )
/////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <assert.h>

#define CITY_PATH_MAX 10

/*****************************************************************/
//Path struct use to represent a path...
struct Path{
    int n; //number of cities
    int cost; // cost of path
    int city_path[CITY_PATH_MAX];
    int visited[CITY_PATH_MAX];
};
typedef struct Path Path;

int num_cities;         //number of cities...
int first_city = 0;     //first city visited, default 0
int **distances;        //city distances...
Path *best_path;        //best path

/****** Function declarations */
void get_input(char filename[], char n[]);       /* Read input from file */

//Path...
void init_path(Path *path);
void init_best_path(Path *path);
void add_city(Path *path, int new_city);
void print_path(Path *path);
int check_best_path(Path *path);
void remove_last(Path *p);
Path* copy_paths(Path *original_path);
void compute_path( Path *p);
/*****************************************************************/

///////////////////////////////////////////////////////////////////
void get_input(char filename[], char n[]){
    FILE * fp;
    int i ,j;  //for the for loops
    
    //attempt to open file...
    fp = fopen(filename, "r");

    //get nume of cities
    //citiesx
    sscanf(n, "%d", &num_cities);

    
    //if file cant be open, do the following...
    if(!fp){
        printf("Cannot open file %s\n", filename);
        exit(1);
    }
    
    
    /* Now, time to allocate the matrices and vectors */
    distances = (int**)malloc(num_cities * sizeof(int*));

    if( !distances){
        printf("Cannot allocate a!\n");
        exit(1);
    }

    for(i = 0; i < num_cities; i++){
        distances[i] = (int *)malloc(num_cities * sizeof(int)); 
        if( !distances[i]){
            printf("Cannot allocate distances[%d]!\n",i);
            exit(1);
        }
    }

    
    ///////////////////////////////////////////////////////////
    /* Now .. Filling the blanks */ //////////////////////////

    for(i = 0; i < num_cities; i++){
        for(j = 0; j < num_cities; j++){
            fscanf(fp,"%d ",&distances[i][j]);
        }
    }

    ////DEBUG//////////////////////////////////////////////////
    for(i = 0; i < num_cities; i++){
        for(j = 0; j < num_cities; j++){
            printf(" %d ", distances[i][j] );
        }
        printf("%s\n"," ");
    }

    //allocate mem space for best_path...
    best_path = malloc(sizeof(Path));

    //close the file....
    fclose(fp);
}

////////////////////////////////////////////////////////////////
void init_path(Path *path){
    int i;
    path->n = 1;                    //set path->n to 1 since we are adding a city
    for(i = 0; i < CITY_PATH_MAX; i++){
        path->city_path[i] = -1;    //initialize city_path elements to -1
        path->visited[i] = 0;       //mark all cities unvisited
    }
    path->city_path[0] = first_city;    //add first_city to path
    path->visited[first_city] = 1;      //mark first city visited
    path->cost = 0;                     //init cost to 0
}

////////////////////////////////////////////////////////////////
void init_best_path(Path *path){
    int i;

    init_path(path);

    path->n = num_cities;

    for(i = 1; i < num_cities; i++){
        path->city_path[i] = i;
        path->cost = path->cost + distances[i-1][i];
        path->visited[i] = 1;
    }
}

////////////////////////////////////////////////////////////////
//adds a city to the current tour...
void add_city(Path *path, int new_city){
    //add distance from last city on path to new city to p->cost
    path->cost = path->cost + distances[path->city_path[path->n-1]][new_city];
    path->city_path[path->n] = new_city;    //add new city to path
    path->visited[new_city] = 1;            //mark visited
    path->n++;                              //increase n
}

////////////////////////////////////////////////////////////////
//prints the path.....
void print_path(Path *path){
    int i;

    printf("%s","Path: " );

    for( i = 0; i < num_cities; i++){
        if(path->city_path[i] != -1){
            printf("%d ", path->city_path[i] );
        }
    }
    printf("%s\n"," " );

    printf("Cost: %d\n", path->cost );
}

////////////////////////////////////////////////////////////////
//return 1 if given tour is the best, otherwise 0
int check_best_path(Path *p){
    int result = 0;

    if(p->cost < best_path->cost){
        result = 1;
    }

    return result;
}

//////////////////////////////////////////////////////////////////
void remove_last(Path *p){
    //decrease p->cost by cost of the distance of the last two cities
    p->cost = p->cost - distances[p->city_path[p->n-2]][p->city_path[p->n-1]];
    p->visited[p->city_path[p->n-1]] = 0;   //mark as unvisited
    p->city_path[p->n-1] = -1;              //set to -1, to mean empty
    p->n--;                                 //decrease n
}
////////////////////////////////////////////////////////////////
// Returns copy of given path
Path* copy_paths(Path *original_path){
    int i;
    Path *new_path;
    new_path = malloc( sizeof( Path));
    init_path(new_path);

    new_path->n = original_path->n;
    new_path->cost = original_path->cost;

    for(i = 0; i < original_path->n; i++){
        new_path->city_path[i] = original_path->city_path[i];
    }

    return new_path;
}

/////////////////////////////////////////////////////////////
//recursive function that computes the path
void compute_path( Path *p ){
    int i;

    if(p->n+1 == num_cities){
    //add remaining city and chekc if it is the best path
        //to add remaing city, we need to find that remaing city..
        for(i = 0; i < num_cities; i++){
            if(p->visited[i] == 0){
                add_city(p, i);
            }
        }
        if(check_best_path(p) == 1){
            Path* curr_best_path;
            //if it is, make copy and replace
            curr_best_path = copy_paths(p);
            //each thread must update best_path one at a time
            #pragma omp critical
            best_path = curr_best_path;
        }

        //remove last two cities
        remove_last(p);
        remove_last(p);

    } else {
        for( i = 0; i < num_cities; i++){
            if(p->visited[i] == 0){
                //add city to path
                add_city(p, i);
                //compute remaining path...
                compute_path(p);
            }
        }
        //remove last city, and mark as unvisited...
        remove_last(p);
    }
}

/************************************************************/
//Main method...
int main(int argc, char *argv[]){   
    int i; 

    //check to see if there are 3 args at the command line...
    if( argc != 3) {
        printf("Usage: tsm filename\n");
        exit(1);
    }
    /* Read the input file and fill the global data structure above */
    //argv[1] = file argv[2] = num of cities 
    get_input(argv[1], argv[2]);

    //initialize best path, to be used for comparison later
    init_best_path(best_path);

    //divide the for loop amongst the threads...
    #pragma omp parallel for
    for(i = 1; i < num_cities;i++){
        //create a path...
        Path *curr_path;
        curr_path = malloc( sizeof( Path));
        init_path(curr_path);

        //add i to said path...
        add_city(curr_path, i);
        //and compute the remainder of the path...
        compute_path(curr_path );
    }

    //print best path!!
    print_path(best_path);



    return 0;
}/***********************************************************/

