#include "driver.h"

double cpuRefSpMV(HostCsrMatrix *M, float *v) {
    answer = (float*) malloc(M->m * sizeof(float));

    struct timeval start, end;
    double elapsed = 0.0;

    /* warmup */
    for(int i = 0; i < 10; i++)
	mkl_scsrgemv((char*)"N", &M->m, M->vals, M->rows, M->cols, v, answer);

    /* count */
    for(int i = 0; i < NITER; i++) {
        gettimeofday (&start, NULL);
        {
            mkl_scsrgemv((char*)"N", &M->m, M->vals, M->rows, M->cols, v, answer);
        }
        gettimeofday (&end, NULL);
        elapsed += (end.tv_sec-start.tv_sec) + 1.e-6*(end.tv_usec - start.tv_usec);
    }

    check_vec(M->m, answer, answer);
    return elapsed / (double) NITER;
}

HostCsrMatrix::HostCsrMatrix(int m, int n, int nnz, int *coo_rows, int *coo_cols, float *coo_vals) :
	CsrMatrix(m, n, nnz, NULL, NULL, NULL)
{
    rows = (int*) malloc((m+1) * sizeof(int));
    cols = (int*) malloc(nnz * sizeof(int));
    vals = (float*) malloc(nnz * sizeof(float));

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
    mkl_scsrcoo(job, &m, vals, cols, rows, &nnz, coo_vals, coo_rows, coo_cols, &info);
    assert(info == 0 && "Converted COO->CSR");
}


