/* Assignment 5 */
#include <stdlib.h>
#include <stdio.h>
#ifdef NO_MKL_LAPACK
#include <lapacke.h>
#else
#include <mkl_lapacke.h>
#endif
#include <math.h>
#include <string.h>

#define SUCCESS 0
#define FAILURE 1

/* For a given active grid point, it gives the index (into vector of values of
   u) of each of its neighbours or -1 if no neighbour */
typedef struct {
    long right;
    long up;
    long left;
    long down;
} BoundaryIndex;

/* For a given active grid point, it gives its (x,y) co-ordinates */
typedef struct {
    long x;
    long y;
} GridPtCoord;

/* how we flatten a 2D array of (both active and inactive) cells into a 1D one */
long gridIndex(long Nx, long Ny, long x, long y){
    return x*Ny+y;
}

typedef struct {
    /* as per spec */
    long           Nx;
    long           Ny;
    long           Na;
    double         Lx;
    double         Ly;
    double         tf;
    double         lambda;
    double         tD;
    /* extended inputs for testing */
    double         lambda2; // scaling for u^3 term
    double         deltaT;
    long           printXIdx; // x co-ord of u to print, -1 for all
    long           printYIdx; // y co-ord of u to print, -1 for all
    long           timeStepInc; /* when running simulation multiple times,
                                   how much to increase num time steps between
                                   multiples of tD each time */
    long           numSims; // number of times to run simulation
    /* next set of fields are derived data */
    long*          gridPtActiveIdx; /* whether a grid point is not
                                       active (-1) or is (holds index
                                       into BoundaryIndex and
                                       GridPtCoord) */
    BoundaryIndex* bndIndex; /* one for each active grid point */
    GridPtCoord*   gridPtCoord;/* one for each active grid point */
    long           nbands; /* bandwith of banded matrix */
} Input;

/* frees memory contained in Input */
void finaliseInput(Input* input){
    free(input->gridPtActiveIdx);
    free(input->bndIndex);
    free(input->gridPtCoord);
}

/* Updates nbands field if absolute value of offset is bigger than existing
   value */
void updateNBands(Input* input, long offset){
    offset = labs(offset);
    if (offset > input->nbands){
        input->nbands = offset;
    }
}

/* populates bndIndex for index i with index of adjoining active cells
   (or -1 if no active cell) and will update BoundaryIndex of any
   adjoining cells. gridPtActiveIdx has a value for each active or inactive
   cell and contains -1 if cell inactive or its index into BoundaryIndex.
   Also updates nbands */
void populateBoundaryIndex(Input* input,
                           long   i){ /* index of this cell in bndIndex */
    /* for ease */
    long*          gridPtActiveIdx = input->gridPtActiveIdx;
    GridPtCoord*   gridPtCoord =   input->gridPtCoord + i;
    BoundaryIndex* bndIndex =   input->bndIndex;
    
    /* Note activeGridIdx is index into active cells so can be used on bndIndex*/
    long activeGridIdx;
    /* see if there is an active cell to the right */
    /* Note gridIdx is index into all (active and non active) cells */
    long gridIdx = gridIndex(input->Nx, input->Ny,
                             gridPtCoord->x+1, gridPtCoord->y);
    if (gridPtCoord->x < input->Nx -1 &&
        (activeGridIdx = gridPtActiveIdx[gridIdx]) != -1){
        bndIndex[activeGridIdx].left = i;
        updateNBands(input, i - activeGridIdx);
    } else {
        activeGridIdx = -1;
    }
    bndIndex[i].right = activeGridIdx;

    /* see if there is an active cell above */
    gridIdx = gridIndex(input->Nx, input->Ny, gridPtCoord->x, gridPtCoord->y+1);
    if (gridPtCoord->y < input->Ny-1 &&
        (activeGridIdx = gridPtActiveIdx[gridIdx]) != -1){
        bndIndex[activeGridIdx].down = i;
        updateNBands(input, i - activeGridIdx);
    } else {
        activeGridIdx = -1;
    }
    bndIndex[i].up = activeGridIdx;
    
    /* see if there is an active cell to the left */
    gridIdx = gridIndex(input->Nx, input->Ny, gridPtCoord->x-1, gridPtCoord->y);
    if (gridPtCoord->x > 0 &&
        (activeGridIdx = gridPtActiveIdx[gridIdx]) != -1){
        bndIndex[activeGridIdx].right = i;
        updateNBands(input, i - activeGridIdx);
    } else {
        activeGridIdx = -1;
    }
    bndIndex[i].left = activeGridIdx;
    
    /* see if there is an active cell below */
    gridIdx = gridIndex(input->Nx, input->Ny, gridPtCoord->x, gridPtCoord->y-1);
    if (gridPtCoord->y > 0 &&
        (activeGridIdx = gridPtActiveIdx[gridIdx]) != -1){
        bndIndex[activeGridIdx].up = i;
        updateNBands(input, i - activeGridIdx);
    } else {
        activeGridIdx = -1;
    }
    bndIndex[i].down = activeGridIdx;
}

/* reads inputs from specified file and populates input. Also reads
   grid values from second specified file. Returns pointer to double
   array containing initial values of u or NULL on failure.
   finaliseInput must always be called on input */
double* readInputs(const char* inputFile,
                   int         readExtraArgs,
                   const char* gridFile,
                   Input*      input){
    int status = FAILURE;
    double* initialU = NULL; /* initial value of u, one for each
                              * active grid point */
    FILE* file = fopen(inputFile, "r");
    if (!file){
        printf("Couldn't open %s\n", inputFile);
        goto done;
    }
    if (fscanf(file, "%ld%ld%ld%lf%lf%lf%lf%lf",
               &input->Nx,&input->Ny, &input->Na, &input->Lx, &input->Ly,
               &input->tf,&input->lambda, &input->tD) != 8){
        printf("Failed to parse %s\n", inputFile);
        goto done;
    }
    input->lambda2 = 1; // always 1 in spec
    input->printXIdx = -1; // switch off by default
    input->printYIdx = -1; // switch off by default
    input->timeStepInc = 0; // default
    input->numSims = 1; // default
    if (readExtraArgs && fscanf(file, "%lf%lf%ld%ld%ld%ld",
                                &input->lambda2,
                                &input->deltaT,
                                &input->printXIdx,
                                &input->printYIdx,
                                &input->timeStepInc,
                                &input->numSims) != 6){
        printf("Failed to parse extra inputs from %s\n", inputFile);
        goto done;
    }
    /* validate inputs */
    if (input->Nx < 2 || input->Ny < 2){
        printf("Require Nx >=2 and Ny >=2\n");
        goto done;
    }
    if (input->Na > input->Nx * input->Ny){
        printf("Number of active grid cells bigger than total "
               "number of cells!\n");
        goto done;
    }
  
    if (input->tf < 0){
        printf("Require tf>= 0\n");
        goto done;
    }
    if (readExtraArgs && input->deltaT > input->tD){
        printf("Require deltaT <= tD\n");
        goto done;
    }
    fclose(file);
    file = fopen(gridFile, "r");
    if (!file){
        printf("Couldn't open %s\n", gridFile);
        goto done;
    }
    if (!(input->gridPtActiveIdx = malloc(input->Nx*input->Ny*sizeof(long))) ||
        !(input->bndIndex = malloc(input->Na * sizeof(BoundaryIndex))) ||
        !(input->gridPtCoord = malloc(input->Na * sizeof(GridPtCoord))) ||
        !(initialU = malloc(input->Na * sizeof(double)))){
        printf("Failed to allocate memory for grid info");
        goto done;
    }
    input->nbands = 0; // we'll calculate this in populateBoundaryIndex
    /* initialise gridPtActiveIdx (has a value for each active or
       inactive cell and contains -1 if cell inactive or, if active,
       its index into BoundaryIndex) */
    for (long i = 0; i < input->Nx * input->Ny; i++){
        input->gridPtActiveIdx[i] = -1; // set all cells inactive
                                        // until otherwise
    }
    for (long i = 0; i < input->Na; i++){
        GridPtCoord* thisGridPtCoord = input->gridPtCoord+i;
        if (fscanf(file, "%ld%ld%lf",
                   &thisGridPtCoord->x,
                   &thisGridPtCoord->y,
                   &initialU[i])!= 3){
            printf("Failed to read %ldth row from %s\n", i+1, gridFile);
            goto done;
        }
        if (thisGridPtCoord->x < 0 || thisGridPtCoord->x >= input->Nx ||
            thisGridPtCoord->y < 0 || thisGridPtCoord->y >= input->Ny){
            printf("Error: row %ld in %s contains information for cell "
                   "(%ld,%ld) but (Nx, Ny) = (%ld, %ld)\n",
                   i, gridFile, thisGridPtCoord->x, thisGridPtCoord->y,
                   input->Nx, input->Ny);
            goto done;
        }
        long gridIdx = gridIndex(input->Nx, input->Ny,
                                 thisGridPtCoord->x, thisGridPtCoord->y);
        if (input->gridPtActiveIdx[gridIdx] != -1){
            printf("Warning: repeated grid point for (%ld,%ld) at %ld'th row\n",
                   thisGridPtCoord->x, thisGridPtCoord->y, i);
        }
        input->gridPtActiveIdx[gridIdx] = i;
        populateBoundaryIndex(input, i);
    }
    status = SUCCESS;
done:
    if (file){
        fclose(file);
    }
    if (status == FAILURE){
        free(initialU);
        initialU = NULL;
    }
    return initialU;
}

void printOutput(FILE*              file,
                 double             time,
                 long               Na,
                 const GridPtCoord* gridPtCoord,
                 double*            u){ /* the solution to the PDE */
    for (long n = 0; n < Na; n++){
        fprintf(file, "%g %ld %ld %g\n",
                time, gridPtCoord[n].x, gridPtCoord[n].y, u[n]);
    }
}

/* version of printOutput to support easier importing numbers into a
   spreadsheet */
void printOutputTransposed(FILE*   file,
                           Input*  input,
                           long    numTimeSteps, // if > 0 then print this
                           double  time,
                           double* u){
    if (input->printXIdx != -1 && input->printYIdx != -1){
        // only displaying single point
        long cellIdx = gridIndex(input->Nx, input->Ny,
                                 input->printXIdx,
                                 input->printYIdx);
        long activeIdx = input->gridPtActiveIdx[cellIdx];
        if (activeIdx == -1){
            fprintf(file, "printOutputTransposed: requested cell not active");
        } else {
            if (numTimeSteps > 0){
                fprintf(file, "%ld, ", numTimeSteps);
            }
            fprintf(file, "%g, ", u[activeIdx]);
        }
    } else if (input->printYIdx != -1){
        // only printing single row, so put entire row
        // on single line
        fprintf(file, "%g", time);
        for (long xIdx = 0; xIdx < input->Nx; xIdx++){
            long cellIdx = gridIndex(input->Nx, input->Ny,
                                     xIdx,
                                     input->printYIdx);
            long activeIdx = input->gridPtActiveIdx[cellIdx];
            if (activeIdx == -1){
                fprintf(file, ", "); // not active so skip
            } else {
                fprintf(file, ", %g", u[activeIdx]);
            }
        }
        fprintf(file, "\n");
    } else if (input->printXIdx != -1){
        // only printing single column, so put entire column
        // on single line
        fprintf(file, "%g", time);
        for (long yIdx = 0; yIdx < input->Ny; yIdx++){
            long cellIdx = gridIndex(input->Nx, input->Ny,
                                     input->printXIdx,
                                     yIdx);
            long activeIdx = input->gridPtActiveIdx[cellIdx];
            if (activeIdx == -1){
                fprintf(file, ", "); // not active so skip
            } else {
                fprintf(file, ", %g", u[activeIdx]);
            }
        }
        fprintf(file, "\n");
    } else {
        fprintf(file, "time %f:\n", time);
        // print matrix of values. First row is x axis.First column is y axis
        for (long xIdx = 0; xIdx < input->Nx; xIdx++){
            fprintf(file, ",%g ", input->Lx/(input->Nx * 2) * (xIdx * 2 + 1));
        }
        for (long yIdx = 0; yIdx < input->Ny; yIdx++){
            fprintf(file, "\n%g", input->Ly/(input->Ny * 2) * (yIdx * 2 + 1));
            for (long xIdx = 0; xIdx < input->Nx; xIdx++){
                long cellIdx = gridIndex(input->Nx, input->Ny,
                                         xIdx, yIdx);
                long activeIdx = input->gridPtActiveIdx[cellIdx];
                if (activeIdx == -1){
                    fprintf(file, ", "); // not active so skip
                } else {
                    fprintf(file, ", %g", u[activeIdx]);
                }
            }
        }
        fprintf(file, "\n");
    }
}

/* start of code copied from band_utility.c */

/* Define structure that holds band matrix information */
typedef struct band_mat{
  long ncol;        /* Number of columns in band matrix */
  long nbrows;      /* Number of rows (bands in original matrix) */
  long nbands_up;   /* Number of bands above diagonal */
  long nbands_low;  /* Number of bands below diagonal */
  double *array;    /* Storage for the matrix in banded format */
  /* Internal temporary storage for solving inverse problem */
  long nbrows_inv;  /* Number of rows of inverse matrix */
  double *array_inv;/* Store the inverse if this is generated */
  int *ipiv;        /* Additional inverse information */
} band_mat;

/* Initialise a band matrix of a certain size, allocate memory,
   and set the parameters.  */ 
int init_band_mat(band_mat *bmat, long nbands_lower, long nbands_upper, long n_columns) {
    bmat->nbrows = nbands_lower + nbands_upper + 1;
    bmat->ncol   = n_columns;
    bmat->nbands_up = nbands_upper;
    bmat->nbands_low= nbands_lower;
    bmat->array      = (double *) malloc(sizeof(double)*bmat->nbrows*bmat->ncol);
    bmat->nbrows_inv = bmat->nbands_up*2 + bmat->nbands_low + 1;
    bmat->array_inv  = (double *)
        calloc(sizeof(double), (bmat->nbrows+bmat->nbands_low)*bmat->ncol);
    bmat->ipiv       = (int *) malloc(sizeof(int)*bmat->ncol);
    if (bmat->array == NULL || bmat->array_inv == NULL || bmat->ipiv ==NULL) {
        return 0;
    }  
    /* Initialise array to zero */
    long i;
    for (i=0;i<bmat->nbrows*bmat->ncol;i++) {
        bmat->array[i] = 0.0;
    }
    return 1;
}

/* Finalise function: should free memory as required */
void finalise_band_mat(band_mat *bmat) {
  free(bmat->array);
  free(bmat->array_inv);
  free(bmat->ipiv);
}

/* Get a pointer to a location in the band matrix, using
   the row and column indexes of the full matrix. Returns NULL if indexes out
   of bounds */
double* getp(band_mat *bmat, long row, long column) {
  int bandno = bmat->nbands_up + row - column;
  if(row<0 || column<0 || row>=bmat->ncol || column>=bmat->ncol ) {
    printf("Indexes out of bounds in getp: %ld %ld %ld \n",
           row,column,bmat->ncol);
    return NULL;
  }
  return &bmat->array[bmat->nbrows*column + bandno];
}

/* Set an element of a band matrix to a desired value based on the pointer
   to a location in the band matrix, using the row and column indexes
   of the full matrix. Returns SUCCESS or FAILURE         */
int setv(band_mat *bmat, long row, long column, double val) {
    double* location = getp(bmat,row,column);
    if (!location){
        return FAILURE;
    }
    *location = val;
    return SUCCESS;
}

/* Solve the equation Ax = b for a matrix a stored in band format
   and x and b real arrays                                          */
int solve_Ax_eq_b(band_mat *bmat, double *x, double *b) {
  /* Copy bmat array into the temporary store */
  int i,bandno;
  for(i=0;i<bmat->ncol;i++) { 
    for (bandno=0;bandno<bmat->nbrows;bandno++) {
      bmat->array_inv[bmat->nbrows_inv*i+(bandno+bmat->nbands_low)] = bmat->array[bmat->nbrows*i+bandno];
    }
    x[i] = b[i];
  }

  long nrhs = 1;
  long ldab = bmat->nbands_low*2 + bmat->nbands_up + 1;
  int info = LAPACKE_dgbsv( LAPACK_COL_MAJOR, bmat->ncol, bmat->nbands_low, bmat->nbands_up, nrhs, bmat->array_inv, ldab, bmat->ipiv, x, bmat->ncol);
  return info;
}

/* end of code copied from band_utility.c */

/* Swap 2 pointers over */
void swapDoublePtr(double** p1, double** p2){
    double* tmp = *p2;
    *p2 = *p1;
    *p1 = tmp;
}

void printUsage(const char* prog){
    printf("Command line parameters not recognised. Specify 0 or 3 or 4\n"
           "Usage: %s [[-t] inputFile gridFile outputFile]\n", prog);
}

/* adjust u before matrix inversion/multiplication (see maths for
   implicit approach for time dimension for PDEs) */
void adjustU(const Input* input,
             double*      u,
             double*      u_tmp, // temporary storage for u
             double       deltaT,
             double       alphaX,
             double       alphaY){
    /* create a copy to work from */
    for (long i = 0; i < input->Na; i++){
        u_tmp[i] = u[i];
    }
    for (long i = 0; i < input->Na; i++){
        // access BoundaryIndex for i'th cell
        BoundaryIndex* bndIndex = input->bndIndex + i;
        // what we're going to add on. Initially the lambda * u and the u^3
        // term
        double adj = deltaT * ((input->lambda/2.0) * u_tmp[i] -
                               input->lambda2 * u_tmp[i] * u_tmp[i] * u_tmp[i]);
        long idx;
        // adjustment for x derivative
        if ((idx = bndIndex->right) != -1){
            adj += alphaX * (u_tmp[idx] - u_tmp[i]);
        }
        if ((idx = bndIndex->left) != -1){
            adj += alphaX * (u_tmp[idx] - u_tmp[i]);
        }
        // adjustment for y derivative
        if ((idx = bndIndex->up) != -1){
            adj += alphaY * (u_tmp[idx] - u_tmp[i]);
        }
        if ((idx = bndIndex->down) != -1){
            adj += alphaY * (u_tmp[idx] - u_tmp[i]);
        }
        u[i] += adj;
    }    
}

int main(int argc, const char* argv[]) {
    /* for error handling, declare everything here that needs to cleaned up */
    int status = FAILURE; /* until otherwise */
    band_mat bmat = {0, 0, 0, 0, NULL, 0, NULL, NULL};
    Input    input = {0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1, -1, 0, 1,
                      NULL, NULL, NULL, 0};
    FILE*    file = NULL;
    double*  u = NULL; /* stores solution to PDE */
    double*  u_next = NULL;
    double*  u_tmp = NULL;
    double*  u_initial = NULL;
    
    /* make testing easier - allow input and output files to be specified
       on command line */
    const char* inputName;
    const char* gridName;
    const char* outputName;
    /** if extensions is true then various extra features are turned on
        including reading extra parameters from input file */
    int extensions= 0; /* default to required output */
    if (argc == 1){
        /* default to normal if not specified */
        inputName = "input.txt";
        gridName = "coefficients.txt";
        outputName = "output.txt";
    } else if (argc == 4 || argc == 5 ){
        if (argc == 5){
            if (strcmp(argv[1], "-t") != 0){
                printUsage(argv[0]);
                goto done;
            }
            extensions = 1;
        }
        inputName = argv[1+extensions];
        gridName = argv[2+extensions];
        outputName = argv[3+extensions];
    } else {
        printUsage(argv[0]);
        goto done;
    }

    u_initial = readInputs(inputName, extensions, gridName, &input);
    if (!u_initial){
        goto done;
    }
    double deltaT = extensions? input.deltaT: input.tD; 
    long numTimeStepsPerPrint = (long) ceil(input.tD/deltaT);
    long numTimeSteps = (long) ceil(input.tf/input.tD) * numTimeStepsPerPrint;
    if (extensions){
        printf("Nx=%ld, Ny=%ld, Na=%ld, Lx=%f, Ly=%f, tf=%f, lambda=%f, "
               "tD=%f, lambda2=%f, deltaT=%f, "
               "printXIdx=%ld, printYIdx=%ld, timeStepInc=%ld, numSims=%ld, "
               "numTimeSteps=%ld\n",
               input.Nx, input.Ny, input.Na, input.Lx,
               input.Ly, input.tf, input.lambda, input.tD,
               input.lambda2, input.deltaT,
               input.printXIdx, input.printYIdx,
               input.timeStepInc, input.numSims, numTimeSteps );
        printf("Active cells:\n");
        printf("Idx\tRight\tup\tleft\tdown\tinitialValue\n");
        for (long n = 0; n < input.Na; n++){
            printf("%ld\t%ld\t%ld\t%ld\t%ld\t%f\n", n,
                   input.bndIndex[n].right, input.bndIndex[n].up,
                   input.bndIndex[n].left, input.bndIndex[n].down,
                   u_initial[n]);
        }
    }
    long ncols = input.Na;
    long nbands_low = input.nbands;  
    long nbands_up  = input.nbands;
    if (init_band_mat(&bmat, nbands_low, nbands_up, ncols) == 0){
        printf("Failed to initialise band_mat\n");
        goto done;
    }
    u = malloc(sizeof(double)*input.Na);
    u_next = malloc(sizeof(double)*input.Na);
    u_tmp = malloc(sizeof(double)*input.Na);
    if (!u || !u_tmp || !u_initial){
        printf("Failed to allocate %ld doubles\n", input.Na);
        goto done;
    }
    double deltaX = input.Lx/(input.Nx);/* gap between grid
                                         * points on x axis */
    double deltaY = input.Ly/(input.Ny);/* gap between grid
                                         * points on y axis */
    /* open file for output */
    file = fopen(outputName, "w");
    if (!file){
        printf("Couldn't open %s for writing\n", outputName);
        goto done;
    }

    if (input.printXIdx != -1 && input.printYIdx != -1){
        // only printing single point, so put values for different
        // time steps in a row. First row is time value for each column
        fprintf(file, ", 0");
        for (long timeIdx = 0; timeIdx * deltaT < input.tf; timeIdx++){
            fprintf(file, ", %g", (timeIdx+1) * deltaT);
        }
        fprintf(file, "\n");
    } else if (input.printYIdx != -1){
        // only printing single row  so put entire row 
        // on single line. First row is x value for each entry
        for (long idx = 0; idx < input.Nx; idx++){
            fprintf(file, ", %g", deltaX * (idx + 0.5));
        }
        fprintf(file, "\n");
    } else if (input.printXIdx != -1){
        // only printing single column, so put entire row or column
        // on single line. First row is y value for each entry
        for (long idx = 0; idx < input.Ny; idx++){
            fprintf(file, ", %g", deltaY * (idx + 0.5));
        }
        fprintf(file, "\n");
    }
    
    /* extend functionality by providing ability to repeat calculation for
       different deltaT */
    for (long simIdx = 0; simIdx < input.numSims; simIdx++){
        // reset u to initial value
        for (long idx = 0; idx < input.Na; idx++){
            u[idx] = u_initial[idx];
        }
        // recalculate deltaT
        if (simIdx > 0){
            numTimeStepsPerPrint += input.timeStepInc;
            numTimeSteps = (long) ceil(input.tf/input.tD) * numTimeStepsPerPrint;
            deltaT = input.tf/(numTimeSteps);
        }
        if (extensions){
            printf("numTimeSteps = %ld, numTimeStepsPerPrint = %ld, "
                   "deltaT = %lf: ",
                   numTimeSteps, numTimeStepsPerPrint, deltaT);
        }
        double alphaX = deltaT/(2.0 * deltaX * deltaX);
        double alphaY = deltaT/(2.0 * deltaY * deltaY);
        /* Set the matrix values equal to the coefficients of the grid
           values. Note that boundaries are treated with special cases */
        /* If we've got the maths right then VERY figuratively speaking,
           what we want for the case of no boundaries is
           *            |  -alpha1             |   
           *   -alpha2  | 1+2alpha1+2alpha2    |   -alpha2
           *            |  -alpha1             |   
           Here the row above/below refers to the other dimension, which because
           we've flattened u into a single column corresponds to other points
           further along the row. For each boundary, junk the alpha term and
           reduce by one the alpha term in the diagonal.
           We also choose to use the implicit scheme for the lambda u term
        */
        for (long i = 0; i < input.Na; i++) {
            /* set up each row which corresponds to how one value of u is
               calculated */
            double diagonalEntry = 1 - input.lambda * deltaT/2.0; //to begin with
            // access BoundaryIndex for i'th cell
            BoundaryIndex* bndIndex = input.bndIndex + i;
            long idx;
            // consider x derivative
            //printf("Row %ld: ", i);
            if ((idx = bndIndex->right) != -1){
                if (setv(&bmat, i, idx, -alphaX) != SUCCESS){
                    goto done;
                }
                //printf("Col %ld = %lf ", idx, -alphaX);
                diagonalEntry += alphaX;
            }
            if ((idx = bndIndex->left) != -1){
                if (setv(&bmat, i, idx, -alphaX) != SUCCESS){
                    goto done;
                }
                //printf("Col %ld = %lf ", idx, -alphaX);
                diagonalEntry += alphaX;
            }
            // repeat for y derivative
            if ((idx = bndIndex->up) != -1){
                if (setv(&bmat, i, idx, -alphaY) != SUCCESS){
                    goto done;
                }
                //printf("Col %ld = %lf ", idx, -alphaY);
                diagonalEntry += alphaY;
            }
            if ((idx = bndIndex->down) != -1){
                if (setv(&bmat, i, idx, -alphaY) != SUCCESS){
                    goto done;
                }
                //printf("Col %ld = %lf ", idx, -alphaY);
                diagonalEntry += alphaY;
            }
            // finally the diagonal
            setv(&bmat, i, i, diagonalEntry);
            //printf("Col %ld = %lf\n", i, diagonalEntry);
        }
   
        /* print initial starting values. The spec says the output is required
           for t = 0, tD, 2tD, ... */
        if (extensions){
            printOutputTransposed(file, &input, numTimeSteps, 0, u);
        } else {
            printOutput(file, 0, input.Na, input.gridPtCoord, u);
        }
        /* now start the time evolution */
        long printIdx = input.tD/deltaT; // 1 if no extensions
        // to do: check tf is always done if tD divides tf
        for (long timeIdx = 0; timeIdx * deltaT < input.tf; timeIdx++){
            /* must adjust u before matrix inversion/multiplication */
            adjustU(&input, u, u_tmp, deltaT, alphaX, alphaY);
            int result = solve_Ax_eq_b(&bmat, u_next, u);
            if (result != 0){
                printf("solve_Ax_eq_b failed with code %d\n", result);
                goto done;
            }
            if ((timeIdx+1) % printIdx == 0){
                if (!extensions){
                    printOutput(file, (timeIdx+1) * deltaT, input.Na,
                                input.gridPtCoord, u_next);
                } else {
                    printOutputTransposed(file, &input, 0, (timeIdx+1) * deltaT,
                                          u_next);
                }
            }
            swapDoublePtr(&u, &u_next);
            if (extensions){
                printf(".");
                fflush(NULL);
            }
        }
        if (extensions){
            printf("\n");
            fprintf(file, "\n");
        }
    }
    status = SUCCESS;
done:
    finalise_band_mat(&bmat);
    finaliseInput(&input);
    if (file){
        fclose(file);
    }
    free(u);
    free(u_initial);
    free(u_next);
    free(u_tmp);
    return status;
}


                
    
    
            
                
