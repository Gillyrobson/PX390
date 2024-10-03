/*******************************************************
* This program solves coupled equations
* 
* To compile, run: gcc -Wall -Werror -std=c99 -lm
*
* List of identified errors:
*  Line       Brief description of a fix
* Number
* -----------------------------------------
* Example (not a real error):
*  21 ...... Removed space between(void) and {
*  16     Missing .h from library header stdlib
*  18     PI not defined to high enough accuracy 
*  37     missing a -1 from current_U[num_grid_points]
*  38     input of 1 instead of 2 in current_U     
*  39     missing line setting current_U[num_grid_points] = current_U[1]
*  46     i starts at 1 not 0
*  47     incorrect formula of next_u, sign change of last term
*  48     incorrect formula of next_v, sign change of last term
*  50     incorrect label of current_u when it should be next_u, missing -1 off num_grid_points
*  51     incorrect label of current_u when it should be next_u, incorrect input of 1
*  52     incorrect label of current_u when it should be next_v, missing -1 off num_grid_points
*  53     incorrect label of current_u when it should be next_v , incorrect input of 1    
*  59     missing declaring tempory pointer tmp as a
*  61     missing value b being assigned to tmp
*  79     value of dt too large, creating instability
*  76     incorrect +1
*  79     initialisation of current time missing starting value
*  88     incorrect memory allocation check, returned the opposite
*  98     missing j=0 and letting j<num_time_steps
*  105    incorrect line, memory has already been allocated
*  106    incorrect line, memory has already been allocated
*******************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define PI 3.14159265358979323846

void PrintCurrent(double current_time,unsigned int num_grid_points, double dx, double* current_U, double* current_V){
    //function to print current state of U and V for a given time step
    for(int i=0; i<num_grid_points; i++){
        double x = dx * i;
        printf("%g %g %g %g\n", current_time, x, current_U[i+1], current_V[i+1]);
    }
}

void Initialise(unsigned int num_grid_points,double dx, double* current_U, double* current_V, double length){
  //function to initialise current U and V to their initial conditions
  
  for(unsigned int i=0; i<num_grid_points; i++){
    double x = dx*i;
    current_U[i+1] = 1.0+sin((2*PI*x)/length);
  }
  current_U[0] = current_U[num_grid_points-1];
  current_U[num_grid_points+1] = current_U[2];
  current_U[num_grid_points] = current_U[1];

}


void CalculateNext(double K, double dt,unsigned int num_grid_points, double dx, double* current_U, double* current_V,  double* next_U, double* next_V){
  //function to calculate next time step of U and V
  
  for(unsigned int i=1; i<num_grid_points+1; i++){
      next_U[i] = current_U[i] + (((K*dt)/(dx*dx)) * (current_U[i+1] + current_U[i-1] - (2*current_U[i]))) + (dt*current_V[i]*(current_U[i]+1));
      next_V[i] = current_V[i] + (((K*dt)/(dx*dx)) * (current_V[i+1] + current_V[i-1] - (2*current_V[i]))) - (dt*current_U[i]*(current_V[i]+1));
  } 
  next_U[0] = next_U[num_grid_points-1];       
  next_U[num_grid_points+1] = next_U[2];     
  next_V[0] = next_V[num_grid_points-1];
  next_V[num_grid_points+1] = next_V[2];
  
}

void MemSwap(double** a, double** b){
    //function to swap two double arrays
    double* tmp = *a;
    *a = *b;
    *b = tmp;
}


int main(){

  // declaring constant K, domain length, number of grid points and final simulation time
  double K = 3.6; 
  double length = 16.873;
  unsigned int num_grid_points = 100;
  double final_time = 1.0;

  //calculating time and length step size and number of time steps
  double dx = length/(num_grid_points-1); // ok 
  double dt = 1.0/400; 
  unsigned int num_time_steps = final_time/dt;

  //initialisation of current time
  double current_time = 0.0; /* presumably 0.0 is correct */

  //allocating current and next step U and V arrays
  double* current_U = malloc((num_grid_points+2)*sizeof(double)); // ok
  double* current_V = calloc(num_grid_points+2, sizeof(double)); // ok
  double* next_U = malloc((num_grid_points+2)*sizeof(double)); // ok
  double* next_V = malloc((num_grid_points+2)*sizeof(double)); // ok

  //check to determine if memory allocation has been performed correctly
  if (!current_U || !current_V || !next_U || !next_V) {
    printf("Memory allocation failed\n");
    return 0;
  }
  
  Initialise(num_grid_points, dx, current_U, current_V, length);
  
  PrintCurrent(current_time, num_grid_points, dx, current_U, current_V);

  //loop over timesteps
  for(unsigned int j = 0; j < num_time_steps; j++){
    
    current_time += dt;
    
    CalculateNext(K, dt, num_grid_points, dx, current_U, current_V, next_U, next_V);

    //making next step current step
    MemSwap(&current_U, &next_U);
    MemSwap(&current_V, &next_V);

    PrintCurrent(current_time, num_grid_points, dx, current_U, current_V);
    
  }

  //memory clean up
  free(current_U);
  free(current_V);
  free(next_U);
  free(next_V);

  return 0;

}
