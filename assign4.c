/* Assignment 4 */
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

/* reads inputs from specified file as well as potential values from second
   specified file. Returns pointer to double array containing potential
   or NULL on failure */
double* readInputs(const char* inputFile,
                   double*     xL,
                   double*     xR,
                   long*       N,
                   double*     aL,
                   double*     aR,
                   double*     bL,
                   double*     bR,
                   double*     E0,
                   long*       numIter,
                   int*        iterPeriod,
                   int*        gridPeriod,
                   int*        normIndex,
                   int         readExtraArgs,
                   const char* potentialFile){
    int status = FAILURE;
    double* potential = NULL;
    FILE* file = fopen(inputFile, "r");
    if (!file){
        printf("Couldn't open %s\n", inputFile);
        goto done;
    }
    /* if (fscanf(file, "%lf\n%lf\n%ld\n%lf\n%lf\n%lf\n%lf\n%lf\n%ld\n",*/
    if (fscanf(file, "%lf%lf%ld%lf%lf%lf%lf%lf%ld",
               /*   */xL,xR, N,aL,aR, bL,bR,E0,numIter) != 9){
        printf("Failed to parse %s\n", inputFile);
        goto done;
    }
    if (readExtraArgs && fscanf(file, "%d%d%d",
                                iterPeriod, gridPeriod, normIndex) != 3){
        printf("Failed to parse extra inputs %s\n", inputFile);
        goto done;
    }
    fclose(file);
    file = fopen(potentialFile, "r");
    if (!file){
        printf("Couldn't open %s\n", potentialFile);
        goto done;
    }
    if (!(potential = malloc(*N*sizeof(double)))){
        printf("Failed to allocate %ld doubles\n", *N);
        goto done;
    }
    for (long i = 0; i < *N; i++){
        if (fscanf(file, "%lf", &potential[i])!= 1){
            printf("Failed to read %ldth double from %s\n", i+1, potentialFile);
            goto done;
        }
    }
    /* validate inputs */
    if (*N < 3){
        printf("Require N>=3\n");
        goto done;
    }
    if ((*aL)*(*aL) + (*bL)*(*bL) == 0){
        printf("Require aL^2 + bL^2 > 0\n");
        goto done;
    }
  
    if ((*aR)*(*aR) + (*bR)*(*bR) == 0){
        printf("Require aR^2 + bR^2 > 0\n");
        goto done;
    }

    if (*xR <= *xL){
        printf("Require xR <= xL\n");
        goto done;
    }
       
    if (readExtraArgs && *normIndex >= *N){
        printf("normIndex (%d) >= N (%ld)\n", *normIndex, *N);
        goto done;
    }
                    
    status = SUCCESS;
done:
    if (file){
        fclose(file);
    }
    if (status == FAILURE){
        free(potential);
        potential = NULL;
    }
    return potential;
}

void printOutput(FILE*   file,
                 long    i,
                 double  xL,
                 double  xR,
                 double  delta,
                 long    N,     /* number of grid points */
                 double* psi){
    for (long n = 0; n < N; n++){
        fprintf(file, "%ld, %g, %g\n", i /* iteration number */,
                xL+delta*n, psi[n]);
    }
}

/** transposed version of printOutput except drop iteration number. Easier
    for importing numbers into a spreadsheet */
void printOutputTransposed(FILE*   file,
                           long    i, /* iteration number */
                           double  xL,
                           double  xR, 
                           double  delta,
                           long    N,     /* number of grid points */
                           double* psi,
                           int     gridPeriod,
                           int     normIndex){
    if (i == 0){
        /* print out grid points */
        for (long n = 0; n < N; n++){
            if (n % gridPeriod == 0){
                fprintf(file, "%g%s", xL+delta*n, n == N-1? "\n": ",");
            }
        }
    }
    double scaling = normIndex < 0? 1.0: (1.0/psi[normIndex]);
    for (long n = 0; n < N; n++){
        if (n % gridPeriod == 0){
            fprintf(file, "%g%s", scaling*psi[n], n == N-1? "\n": ",");
        }
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
   the row and column indexes of the full matrix.           */
double *getp(band_mat *bmat, long row, long column) {
  int bandno = bmat->nbands_up + row - column;
  if(row<0 || column<0 || row>=bmat->ncol || column>=bmat->ncol ) {
    printf("Indexes out of bounds in getp: %ld %ld %ld \n",row,column,bmat->ncol);
    exit(1);
  }
  return &bmat->array[bmat->nbrows*column + bandno];
}

/* Retrun the value of a location in the band matrix, using
   the row and column indexes of the full matrix.           */
double getv(band_mat *bmat, long row, long column) {
  return *getp(bmat,row,column);
}

/* Set an element of a band matrix to a desired value based on the pointer
   to a location in the band matrix, using the row and column indexes
   of the full matrix.           */
double setv(band_mat *bmat, long row, long column, double val) {
    *getp(bmat,row,column) = val;
    return val;
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
           "Usage: %s [[-tP,Q,R] inputFile potentialFile outputFile]\n"
           "with integers P, Q, R detailing iterPeriod, gridPeriod, and "
           "normIndex for transposed output\n", prog);
}

int main(int argc, const char* argv[]) {
    /* for error handling, declare everything here that needs to cleaned up */
    int status = FAILURE; /* until otherwise */
    band_mat bmat = {0, 0, 0, 0, NULL, 0, NULL, NULL};
    double* psi = NULL;
    double* psi_next = NULL;
    double* potential = NULL;
    FILE* file = NULL;
    
    /* make testing easier - allow input and output files to be specified
       on command line */
    const char* inputName;
    const char* potentialName;
    const char* outputName;
    int printTransposed = 0; /* default to required output */
    int iterPeriod = 1; /* print out results every iterPeriod */
    int gridPeriod = 1; /* print out subset of psi (every gridPeriod) */
    int normIndex = -1; /* normalise psi such that psi[normIndex] = 1
                           or -1 to not normalise */
    if (argc == 1){
        /* default to normal if not specified */
        inputName = "input.txt";
        potentialName = "potential.txt";
        outputName = "output.txt";
    } else if (argc == 4 || argc == 5 ){
        if (argc == 5){
            if (strcmp(argv[1], "-t") != 0){
                printUsage(argv[0]);
                goto done;
            }
            printTransposed = 1;
        }
        inputName = argv[1+printTransposed];
        potentialName = argv[2+printTransposed];
        outputName = argv[3+printTransposed];
    } else {
        printUsage(argv[0]);
        goto done;
    }
    
    double xL, xR, aL, aR, bL, bR, E0;
    long N, numIter;
    potential = readInputs(inputName, &xL, &xR, &N, &aL, &aR,
                           &bL, &bR, &E0, &numIter,
                           &iterPeriod, &gridPeriod, &normIndex,
                           printTransposed,
                           potentialName);
    if (!potential){
        goto done;
    }
    if (printTransposed){
        printf("xL=%f, xR=%f, aL=%f, aR=%f, bL=%f, bR=%f, E0=%f,"
               "N=%ld,I=%ld\n", xL, xR, aL, aR, bL, bR, E0, N, numIter);
        printf("Potential: ");
        for (long n = 0; n < N; n++){
            printf("%f ", potential[n]);
        }
        printf("\n");
        printf("printTransposed=%i, iterPeriod=%i, "
               "gridPeriod=%i, normIndex=%i\n",
               printTransposed, iterPeriod, gridPeriod, normIndex);
    }
    int dirichletL = bL == 0.0? 1: 0;
    int dirichletR = bR == 0.0? 1: 0;
    /* For a dirichlet boundary condition, we exclude that point from the
       numerical machinery and force psi to be 0 there */
    long ncols = N - dirichletL - dirichletR;
    /* We have a three-point stencil (domain of numerical dependence) of
       our finite-difference equations:
       1 point to the left  -> nbands_low = 1
       1       to the right -> nbands_up  = 1
    */
    long nbands_low = 1;  
    long nbands_up  = 1;
    if (init_band_mat(&bmat, nbands_low, nbands_up, ncols) == 0){
        printf("Failed to initialise band_mat\n");
        goto done;
    }
    psi = malloc(sizeof(double)*N);
    psi_next = malloc(sizeof(double)*N);
    if (!psi || !psi_next){
        printf("Failed to allocate %ld doubles\n", ncols);
        goto done;
    }
    double  delta = (xR - xL)/(N-1); /* gap between grid points */
    double  deltaSq = delta * delta;
    /* Set the matrix values equal to the coefficients of the grid
       values. Note that boundaries are treated with special cases */
    for(long i=0; i<ncols; i++) {
        if (i > 0) {
            setv(&bmat, i, i-1, ((!dirichletR && i == ncols-1)? -2.0: -1.0)/
                 deltaSq);
        }
        if (!dirichletL && i == 0){
            setv(&bmat, i, i,  (2.0*(1.0-aL*delta/bL))/deltaSq -
                 E0+ potential[i+dirichletL]);
        } else if (!dirichletR && i == ncols-1){
            setv(&bmat, i, i,  (2.0*(1.0+aR*delta/bR))/deltaSq -
                 E0+ potential[i+dirichletL]);
        } else {
            setv(&bmat, i, i,  2.0/deltaSq-E0+ potential[i+dirichletL]);
        }
        if (i < ncols-1) {
            setv(&bmat, i, i+1, ((!dirichletL && i == 0)? -2.0: -1.0)/
                 deltaSq);
        }
        /* create initial guess of psi0 */
        psi[i+dirichletL] = 1.0; 
    }
    /* populate rest of psi for case where bL = 0 or bR = 0 */
    if (dirichletL){
        psi[0] = 1.0;
    }
    if (dirichletR){
        psi[N-1] = 1.0;
    }
    
    /* open file for output */
    file = fopen(outputName, "w");
    if (!file){
        printf("Couldn't open %s for writing\n", outputName);
        goto done;
    }
                         
    /* print initial starting guess. The spec says the output is required
       for i in [0, I] which includes i = 0 */
    if (printTransposed){
        printOutputTransposed(file, 0, xL, xR, delta, N, psi,
                              gridPeriod, normIndex);
    } else {
        printOutput(file, 0, xL, xR, delta, N, psi);
    }
    /* enforce boundary conditions if bL or bR = 0 */
    if (dirichletL){
        psi[0] = 0.0;
        psi_next[0] = 0.0; /* we don't calculate psi_next[0], so must set it */
    }
    if (dirichletR){
        psi[N-1] = 0.0;
        psi_next[N-1] = 0.0; /* as per psi_next[0] */
    }
    /* now start the iteration */
    for (long i = 0; i < numIter; i++){
        int result = solve_Ax_eq_b(&bmat, psi_next+dirichletL, psi+dirichletL);
        if (result != 0){
            printf("solve_Ax_eq_b failed with code %d\n", result);
            goto done;
        }
        if (!printTransposed){
            printOutput(file, i+1, xL, xR, delta, N, psi_next);
        } else if ((i+1) % iterPeriod == 0){
            printOutputTransposed(file, i+1, xL, xR, delta, N, psi_next,
                                  gridPeriod, normIndex);
        }
        swapDoublePtr(&psi, &psi_next);
    }
    
    status = SUCCESS;
done:
    finalise_band_mat(&bmat);
    if (file){
        fclose(file);
    }
    free(potential);
    free(psi);
    free(psi_next);
    return status;
}


                
    
    
            
                
