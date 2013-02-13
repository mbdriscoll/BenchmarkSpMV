#include "driver.h"

#pragma offload_attribute (push,target(mic))
static int *Mrows, *Mcols;
static float *Mvals;
#pragma offload_attribute (pop)

#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)
#define TEMP alloc_if(1) free_if(1)

double micRefSpMV(HostCsrMatrix *M, float *v) {
    int m = M->m,
        n = M->n,
        nnz = M->nnz;
    float *actual = (float*) malloc(m * sizeof(float));

    #pragma offload target(mic) \
        in    (v:      length(M->n) ALLOC) \
        nocopy(actual: length(M->m) ALLOC)
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

#if 0
    #pragma offload target(mic) \
        out   (actual: length(m)   FREE) \
        nocopy(v:      length(n)   FREE) \
        nocopy(Mrows:  length(m+1) FREE) \
        nocopy(Mcols:  length(nnz) FREE) \
        nocopy(Mvals:  length(nnz) FREE)
    {}
#endif

    check_vec(M->m, answer, actual);
    return elapsed / (double) NITER;
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

    /* Hack: these will be device pointers only, but they have to have different values with
     * enough space in between so the runtime doesn't complain about overlap. */
    Mrows = coo_rows;
    Mcols = coo_cols;
    Mvals = coo_vals;

#pragma offload target(mic) \
    in(coo_rows:  length(nnz) alloc_if(1) free_if(1)) \
    in(coo_cols:  length(nnz) alloc_if(1) free_if(1)) \
    in(coo_vals:  length(nnz) alloc_if(1) free_if(1)) \
    nocopy(Mrows: length(m+1) alloc_if(1) free_if(0)) \
    nocopy(Mcols: length(nnz) alloc_if(1) free_if(0)) \
    nocopy(Mvals: length(nnz) alloc_if(1) free_if(0))
    mkl_scsrcoo(job, &m, Mvals, Mcols, Mrows, &nnz, coo_vals, coo_rows, coo_cols, &info);

    assert(info == 0 && "Converted COO->CSR on MIC");
}
