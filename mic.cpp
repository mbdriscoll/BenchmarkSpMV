#include "driver.h"

static int *Mrows, *Mcols;
static float *Mvals;

#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)
#define TEMP alloc_if(1) free_if(1)

double micRefSpMV(DeviceCsrMatrix *M, float *v) {
    int m = M->m,
        n = M->n,
        nnz = M->nnz;
    float *actual = (float*) malloc(m * sizeof(float));

    #pragma offload target(mic) \
        in    (v:      length(M->n) align(64) ALLOC) \
        nocopy(actual: length(M->m) align(64) ALLOC)
    {}

    struct timeval start, end;
    double elapsed = 0.0;
    for(int i = 0; i < NITER; i++) {
        gettimeofday (&start, NULL);
        {
            #pragma offload target(mic) \
	        nocopy(Mrows:  length(m+1) REUSE) \
	        nocopy(Mcols:  length(nnz) REUSE) \
	        nocopy(Mvals:  length(nnz) REUSE) \
	        nocopy(actual: length(m)   REUSE) \
	        nocopy(v:      length(n)   REUSE)
            mkl_scsrgemv((char*)"N", &m, Mvals, Mrows, Mcols, v, actual);
        }
        gettimeofday (&end, NULL);
        elapsed += (end.tv_sec-start.tv_sec) + 1.e-6*(end.tv_usec - start.tv_usec);
    }

    #pragma offload target(mic) \
        out   (actual: length(m)   FREE) \
        nocopy(v:      length(n)   FREE) \
        nocopy(Mrows:  length(m+1) FREE) \
        nocopy(Mcols:  length(nnz) FREE) \
        nocopy(Mvals:  length(nnz) FREE)
    {}

    check_vec(M->m, answer, actual);
    return elapsed / (double) NITER;
}

DeviceCsrMatrix::~DeviceCsrMatrix() {
    free(Mrows);
    free(Mcols);
    free(Mvals);
}

DeviceCsrMatrix::DeviceCsrMatrix(int m, int n, int nnz, int *coo_rows, int *coo_cols, float *coo_vals) :
    CsrMatrix(m, n, nnz, NULL, NULL, NULL) {

    /* Use MKL to convert and sort to CSR matrix. */
    int job[] = {
        2, // job(1)=2 (coo->csr with sorting)
        1, // job(2)=1 (one-based indexing for csr matrix)
        1, // job(3)=1 (one-based indexing for coo matrix)
        0, // empty
        nnz, // job(5)=nnz (sets nnz for csr matrix)
        0  // job(6)=0 (all output arrays filled)
    };

    int info;

    /* do i need to malloc host memory? */
    Mrows = (int*) malloc((m+1) * sizeof(int));
    Mcols = (int*) malloc(nnz * sizeof(int));;
    Mvals = (float*) malloc(nnz * sizeof(float));;

#pragma offload target(mic) \
    in(coo_rows:  length(nnz) TEMP) \
    in(coo_cols:  length(nnz) TEMP) \
    in(coo_vals:  length(nnz) TEMP) \
    nocopy(Mrows: length(m+1) align(64) ALLOC) \
    nocopy(Mcols: length(nnz) align(64) ALLOC) \
    nocopy(Mvals: length(nnz) align(64) ALLOC)
    mkl_scsrcoo(job, &m, Mvals, Mcols, Mrows, &nnz, coo_vals, coo_rows, coo_cols, &info);

    assert(info == 0 && "Converted COO->CSR on MIC");
}
