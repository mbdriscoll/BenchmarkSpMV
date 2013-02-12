#include "driver.h"

double micRefSpMV(HostCsrMatrix *M, float *v) {
    answer = (float*) malloc(M->m * sizeof(float));

    struct timeval start, end;
    double elapsed = 0.0;
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

DeviceCsrMatrix::DeviceCsrMatrix(int m, int n, int nnz, int *h_rows, int *h_cols, float *h_vals) :
    CsrMatrix(m, n, nnz, NULL, NULL, NULL) {
/*
        cudaMalloc(&rows, (m+1) * sizeof(int));
        cudaMalloc(&cols, nnz * sizeof(int));
        cudaMalloc(&vals, nnz * sizeof(float));

        cudaMemcpy(rows, h_rows, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cols, h_cols, nnz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(vals, h_vals, nnz*sizeof(float), cudaMemcpyHostToDevice);

        cusparseCreateMatDescr(&desc);
        cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ONE);
*/
}

DeviceHybMatrix::DeviceHybMatrix(DeviceCsrMatrix *dM) :
    CsrMatrix(dM->m, dM->n, dM->nnz, NULL, NULL, NULL) {
/*
        cusparseCreateMatDescr(&desc);
        cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ONE);

        cusparseCreateHybMat(&hybM);
        cusparseScsr2hyb(handle, dM->m, dM->n, dM->desc, dM->vals, dM->rows, dM->cols, hybM,
                HYBLEN, CUSPARSE_HYB_PARTITION_USER);
*/
}

