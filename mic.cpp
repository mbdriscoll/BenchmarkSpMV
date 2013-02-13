#include "driver.h"

double micRefSpMV(HostCsrMatrix *M, float *v) {
    answer = (float*) malloc(M->m * sizeof(float));

    #pragma offload target(mic) \
        in(v: length(M->n) alloc_if(1) free_if(0))
    {}

    struct timeval start, end;
    double elapsed = 0.0;
    for(int i = 0; i < NITER; i++) {
        gettimeofday (&start, NULL);
        {
            int     Mm = M->m,
                *Mrows = M->rows,
                *Mcols = M->cols;
            float *Mvals = M->vals;
            #pragma offload target(mic) \
	        nocopy(Mrows:  length(M->m+1) alloc_if(0) free_if(0)) \
	        nocopy(Mcols:  length(M->nnz) alloc_if(0) free_if(0)) \
	        nocopy(Mvals:  length(M->nnz) alloc_if(0) free_if(0)) \
	        nocopy(answer: length(M->n)   alloc_if(0) free_if(0)) \
	        nocopy(v:      length(M->n)   alloc_if(0) free_if(0))
            mkl_scsrgemv((char*)"N", &Mm, Mvals, Mrows, Mcols, v, answer);
        }
        gettimeofday (&end, NULL);
        elapsed += (end.tv_sec-start.tv_sec) + 1.e-6*(end.tv_usec - start.tv_usec);
    }

    #pragma offload target(mic) \
        out   (answer: length(M->n) alloc_if(0) free_if(1)) \
        nocopy(v:      length(M->m) alloc_if(0) free_if(1))
    {}

    check_vec(M->m, answer, answer);
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

    /* compiler throws internal error if we just put a bare "rows", etc, in the pragma */
    int* thisrows = this->rows;
    int* thiscols = this->cols;
    float* thisvals = this->vals;

#pragma offload target(mic) \
    in(coo_rows:     length(nnz) alloc_if(1) free_if(1)) \
    in(coo_cols:     length(nnz) alloc_if(1) free_if(1)) \
    in(coo_vals:     length(nnz) alloc_if(1) free_if(1)) \
    nocopy(thisrows: length(m+1) alloc_if(1) free_if(0)) \
    nocopy(thiscols: length(nnz) alloc_if(1) free_if(0)) \
    nocopy(thisvals: length(nnz) alloc_if(1) free_if(0))
    mkl_scsrcoo(job, &m, vals, cols, rows, &nnz, coo_vals, coo_rows, coo_cols, &info);

    assert(info == 0 && "Converted COO->CSR on MIC");
}
