#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <sys/time.h>
#include <mkl_spblas.h>
#include "extra/mmio.h"

/* The number of iterations used to get average performance */
#define NITER 1000

/* peak processor constants */
#define CPU_MEMBW_GBS (32.0)
#define MIC_MEMBW_GBS (320.0)
#define CPU_GFLOPS (3.33*6.0)
#define MIC_GFLOPS (1.053*60.0)

/* convenience macros */
#define ALLOC alloc_if(1) free_if(0)
#define FREE  alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)
#define TEMP  alloc_if(1) free_if(1)


/**
 * Reads a MatrixMarket file and sets the input pointers
 */
void mm_read(char* filename, int *m, int *n, int *nnz, int **rowptrs, int **colinds, float **vals) {
    // open file
    FILE* mmfile = fopen(filename, "r");
    assert(mmfile != NULL && "Read matrix file.");

    // read MatrixMarket header
    int status;
    MM_typecode matcode;
    status = mm_read_banner(mmfile, &matcode);
    assert(status == 0 && "Parsed banner.");
    assert(mm_is_matrix(matcode) && mm_is_sparse(matcode) && mm_is_real(matcode));

    // read matrix dimensions
    status = mm_read_mtx_crd_size(mmfile, m, n, nnz);
    assert(status == 0 && "Parsed matrix m, n, and nnz.");
    printf("- matrix is %d-by-%d with %d nnz.\n", *m, *n, *nnz);

    // alloc space for COO matrix
    int *coo_rows = (int*) malloc(*nnz * sizeof(int));
    int *coo_cols = (int*) malloc(*nnz * sizeof(int));
    float *coo_vals = (float*) malloc(*nnz * sizeof(float));

    // read COO values
    for (int i = 0; i < *nnz; i++)
        status = fscanf(mmfile, "%d %d %g\n", &coo_rows[i], &coo_cols[i], &coo_vals[i]);

    // alloc space for CSR matrix
    *rowptrs = (int*) malloc((*m+1)*sizeof(int));
    *colinds = (int*) malloc(*nnz*sizeof(int));
    *vals = (float*) malloc(*nnz*sizeof(int));

    // convert to CSR matrix
    int info;
    int job[] = {
        2, // job(1)=2 (coo->csr with sorting)
        1, // job(2)=1 (one-based indexing for csr matrix)
        1, // job(3)=1 (one-based indexing for coo matrix)
        0, // empty
        *nnz, // job(5)=nnz (sets nnz for csr matrix)
        0  // job(6)=0 (all output arrays filled)
    };
    mkl_scsrcoo(job, m, *vals, *colinds, *rowptrs, nnz, coo_vals, coo_rows, coo_cols, &info);
    assert(info == 0 && "Converted COO->CSR");

    // free COO matrix
    free(coo_rows);
    free(coo_cols);
    free(coo_vals);
}


/**
 * Prints a warning if any of the first N elements of the two vectors disagree.
 */
void check_vec(int n, float *expected, float *actual) {
    int errors = 0;
    for(int i = 0; i < n; i++)
        if ( fabs(expected[i] - actual[i]) > 2.0*FLT_EPSILON )
            errors += 1;

    if (errors)
        fprintf(stderr, "WARNING: Found %d/%d errors in answer.\n", errors, n);
}


/**
 * Allocates and populates a n-vector with random numbers.
 */
float * randvec(int n) {
    float *v = (float*) malloc(n * sizeof(float));
    for(int i = 0; i < n; i++)
        v[i] = rand() / (float) RAND_MAX;
    return v;
}


/**
 * The main method: parse matrices, run benchmarks, print results.
 */
int main(int argc, char* argv[]) {
    // require filename for matrix to test
    if (argc != 2) {
        fprintf(stderr, "Usage: %s MATRIX.mm\n", argv[0]);
        exit(1);
    }

    // -----------------------------------------------------------------------
    // Allocate space for matrices and vectors

    // CSR matrix dimensions and values
    int m, n, nnz;
    int *rowptrs, *colinds;
    float *vals;

    // read matrix from file
    mm_read(argv[1], &m, &n, &nnz, &rowptrs, &colinds, &vals);

    // allocate vectors for computation
    float *v = randvec(n);
    float *cpu_answer = (float*) malloc(m*sizeof(float)),
          *mic_answer = (float*) malloc(m*sizeof(float));

    // -----------------------------------------------------------------------
    // Benchmark MKL performance on CPU

    struct timeval start, end;

    double cpuAvgTimeInSec = 0.0;
    for(int i = 0; i < NITER; i++) {
        gettimeofday (&start, NULL);
        {
            mkl_scsrgemv((char*)"N", &m, vals, rowptrs, colinds, v, cpu_answer);
        }
        gettimeofday (&end, NULL);
        cpuAvgTimeInSec += (end.tv_sec-start.tv_sec) + 1.e-6*(end.tv_usec - start.tv_usec);
    }
    cpuAvgTimeInSec /= (double) NITER;
    
    // -----------------------------------------------------------------------
    // Benchmark MKL performance on MIC

    // offload data to device
    #pragma offload target(mic) \
        in    (v:           length(n)   align(64) ALLOC) \
        in    (rowptrs:     length(m+1) align(64) ALLOC) \
        in    (colinds:     length(nnz) align(64) ALLOC) \
        in    (vals:        length(nnz) align(64) ALLOC) \
        nocopy(mic_answer:  length(m)   align(64) ALLOC)
    {}

    // run benchmark
    double micAvgTimeInSec = 0.0;
    for(int i = 0; i < NITER; i++) {
        gettimeofday (&start, NULL);
        {
            #pragma offload target(mic) \
	        nocopy(rowptrs:    length(m+1) REUSE) \
	        nocopy(colinds:    length(nnz) REUSE) \
	        nocopy(vals:       length(nnz) REUSE) \
	        nocopy(v:          length(m)   REUSE) \
	        nocopy(mic_answer: length(n)   REUSE)
            mkl_scsrgemv((char*)"N", &m, vals, rowptrs, colinds, v, mic_answer);
        }
        gettimeofday (&end, NULL);
        micAvgTimeInSec += (end.tv_sec-start.tv_sec) + 1.e-6*(end.tv_usec - start.tv_usec);
    }
    micAvgTimeInSec /= (double) NITER;

    // copy answer back and cleanup
    #pragma offload target(mic) \
        nocopy(v:          length(n)   FREE) \
        nocopy(rowptrs:    length(m+1) FREE) \
        nocopy(colinds:    length(nnz) FREE) \
        nocopy(vals:       length(nnz) FREE) \
        out   (mic_answer: length(m)   FREE)
    {}

    // -----------------------------------------------------------------------
    // Verify and display results

    // check the solution
    check_vec(m, cpu_answer, mic_answer);

    // calculate total bytes and flops
    double gflop = 2.e-9 * 2.0 * nnz;
    double gbytes = 2.e-9 * (
            nnz * sizeof(float) + // vals
            nnz * sizeof(int) + // cols
            (m+1) * sizeof(int) + // rows
            (n+m) * sizeof(float)); // vectors

    // display results
    printf("Platform  Time         Gflops/s    %%peak Gbytes/s     %%peak\n");
    printf("MKL-host % 1.8f  % 2.8f  %02.f   %02.8f   %02.f\n", cpuAvgTimeInSec,
            gflop/cpuAvgTimeInSec, 100.0*gflop/cpuAvgTimeInSec/CPU_GFLOPS,
            gbytes/cpuAvgTimeInSec, 100.0*gbytes/cpuAvgTimeInSec/CPU_MEMBW_GBS);
    printf("MKL-mic  % 1.8f  % 2.8f  %02.f   %02.8f   %02.f\n", micAvgTimeInSec,
            gflop/micAvgTimeInSec, 100.0*gflop/micAvgTimeInSec/MIC_GFLOPS,
            gbytes/micAvgTimeInSec, 100.0*gbytes/micAvgTimeInSec/MIC_MEMBW_GBS);

    return 0;
}
