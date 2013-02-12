#include "driver.h"

float *answer = NULL;

void
mm_read(char* filename, HostCsrMatrix **hM, DeviceCsrMatrix **dM) {
    FILE* mmfile = fopen(filename, "r");
    assert(mmfile != NULL && "Read matrix file.");

    int status;
    MM_typecode matcode;
    status = mm_read_banner(mmfile, &matcode);
    assert(status == 0 && "Parsed banner.");

    assert( mm_is_matrix(matcode) );
    assert( mm_is_sparse(matcode) );
    assert( mm_is_real  (matcode) );

    int m, n, nnz;
    status = mm_read_mtx_crd_size(mmfile, &m, &n, &nnz);
    assert(status == 0 && "Parsed matrix m, n, and nnz.");
    printf("- matrix is %d-by-%d with %d nnz.\n", m, n, nnz);

    int *coo_rows = (int*) malloc(nnz * sizeof(int));
    int *coo_cols = (int*) malloc(nnz * sizeof(int));
    float *coo_vals = (float*) malloc(nnz * sizeof(float));

    int *csr_rows = (int*) malloc((m+1) * sizeof(int));
    int *csr_cols = (int*) malloc(nnz * sizeof(int));
    float *csr_vals = (float*) malloc(nnz * sizeof(float));

    for (int i = 0; i < nnz; i++)
        status = fscanf(mmfile, "%d %d %g\n", &coo_rows[i], &coo_cols[i], &coo_vals[i]);

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
    mkl_scsrcoo(job, &m, csr_vals, csr_cols, csr_rows, &nnz, coo_vals, coo_rows, coo_cols, &info);
    assert(info == 0 && "Converted COO->CSR");

    *hM = new HostCsrMatrix(m, n, nnz, csr_rows, csr_cols, csr_vals);
    *dM = new DeviceCsrMatrix(m, n, nnz, csr_rows, csr_cols, csr_vals);

    free(coo_rows);
    free(coo_cols);
    free(coo_vals);
    /* don't free csr_rows etc. keep for hM to use */
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

float *
randvec(int n) {
    float *v = (float*) malloc(n * sizeof(float));
    for(int i = 0; i < n; i++)
        v[i] = rand() / (float) RAND_MAX;
    return v;
}

void
check_vec(int n, float *expected, float *actual) {
    int errors = 0;
    for(int i = 0; i < n; i++)
        if ( fabs(expected[i] - actual[i]) > 2.0*FLT_EPSILON )
            errors += 1;

    if (errors)
        fprintf(stderr, "Found %d/%d errors in answer.\n", errors, n);
}

double cpuRefSpMV(HostCsrMatrix *M, float *v) {
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

double gpuRefSpMV(DeviceCsrMatrix *M, float *v_in) {
    float *dv_in, *dv_out;
/*
    cudaMalloc(&dv_in, M->n * sizeof(float));
    cudaMalloc(&dv_out, M->m * sizeof(float));
    cudaMemcpy(dv_in, v_in, M->n * sizeof(float), cudaMemcpyHostToDevice);

    cusparseStatus_t status;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float alpha = 1.0,
          beta = 0.0;

    double elapsed = 0.0;
    struct timeval start, end;
    for(int i = 0; i < NITER; i++) {
        gettimeofday (&start, NULL);
        {
            status = cusparseScsrmv(handle, op, M->m, M->n, M->nnz, &alpha, M->desc,
                    M->vals, M->rows, M->cols, dv_in, &beta, dv_out);
            cudaThreadSynchronize();
        }
        gettimeofday (&end, NULL);
        elapsed += (end.tv_sec-start.tv_sec) + 1.e-6*(end.tv_usec - start.tv_usec);
    }

        assert(status == CUSPARSE_STATUS_SUCCESS);

    float *v_out = (float*) malloc(M->m * sizeof(float));
    cudaMemcpy(v_out, dv_out, M->m * sizeof(float), cudaMemcpyDeviceToHost);
    check_vec(M->m, answer, v_out);

    return elapsed / (double) NITER;
*/
    return 1.0;
}

double hybRefSpMV(DeviceHybMatrix *dM, float *v_in) {
    float *dv_in, *dv_out;
/*
    cudaMalloc(&dv_in, dM->n * sizeof(float));
    cudaMalloc(&dv_out, dM->m * sizeof(float));
    cudaMemcpy(dv_in, v_in, dM->n * sizeof(float), cudaMemcpyHostToDevice);

    cusparseStatus_t status;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float alpha = 1.0,
          beta = 0.0;

    double elapsed = 0.0;
    struct timeval start, end;
    for(int i = 0; i < NITER; i++) {
        gettimeofday (&start, NULL);
        {
            status = cusparseShybmv(handle, op, &alpha, dM->desc, dM->hybM, dv_in, &beta, dv_out);
            cudaThreadSynchronize();
        }
        gettimeofday (&end, NULL);
        elapsed += (end.tv_sec-start.tv_sec) + 1.e-6*(end.tv_usec - start.tv_usec);
    }

        assert(status == CUSPARSE_STATUS_SUCCESS);

    float *v_out = (float*) malloc(dM->m * sizeof(float));
    cudaMemcpy(v_out, dv_out, dM->m * sizeof(float), cudaMemcpyDeviceToHost);
    check_vec(dM->m, answer, v_out);

    return elapsed / (double) NITER;
*/
    return 1.0;
}

double MyGpuSpMV(DeviceCsrMatrix *M, float *v_in) {
    float *dv_in, *dv_out;
/*
    cudaMalloc(&dv_in, M->n * sizeof(float));
    cudaMalloc(&dv_out, M->m * sizeof(float));
    cudaMemcpy(dv_in, v_in, M->n * sizeof(float), cudaMemcpyHostToDevice);

    cusparseStatus_t status;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float alpha = 1.0,
          beta = 0.0;

    double elapsed = 0.0;
    struct timeval start, end;
    for(int i = 0; i < NITER; i++) {
        gettimeofday (&start, NULL);
        {
            status = my_cusparseScsrmv(handle, op, M->m, M->n, M->nnz, &alpha, M->desc,
                    M->vals, M->rows, M->cols, dv_in, &beta, dv_out);
            cudaThreadSynchronize();
        }
        gettimeofday (&end, NULL);
        elapsed += (end.tv_sec-start.tv_sec) + 1.e-6*(end.tv_usec - start.tv_usec);
    }

    assert(status == CUSPARSE_STATUS_SUCCESS);

    float *v_out = (float*) malloc(M->m * sizeof(float));
    cudaMemcpy(v_out, dv_out, M->m * sizeof(float), cudaMemcpyDeviceToHost);
    check_vec(M->m, answer, v_out);

    return elapsed / (double) NITER;
*/
    return 1.0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s MATRIX.mm\n", argv[0]);
        exit(1);
    }

#ifdef __INTEL_OFFLOAD
    printf("Checking for number of Intel Xeon Phi devices...");
    int num_devices = _Offload_number_of_devices();
    printf(" %d\n",num_devices);
#endif

    printf("Reading matrix at %s.\n", argv[1]);
    HostCsrMatrix *hM;
    DeviceCsrMatrix *dM;
    mm_read(argv[1], &hM, &dM);
    DeviceHybMatrix *hybM = new DeviceHybMatrix(dM);

    float *v = randvec(hM->n);

    double cpuRefTime = cpuRefSpMV(hM, v); // warm cache
           cpuRefTime = cpuRefSpMV(hM, v);
    double micRefTime = micRefSpMV(hM, v); // warm cache
           micRefTime = micRefSpMV(hM, v);

    double gflop = 2.e-9 * 2.0 * hM->nnz;
    double gbytes = 2.e-9 * (
            hM->nnz * sizeof(float) + // vals
            hM->nnz * sizeof(int) + // cols
            hM->m * sizeof(int) + // rows
            (hM->n + hM->m) * sizeof(float)); // vectors

    printf("Platform  Time         Gflops/s    %%peak Gbytes/s     %%peak\n");
    printf("MKL-host % 1.8f  % 2.8f  %02.f   %02.8f   %02.f\n", cpuRefTime,
            gflop/cpuRefTime, 100.0*gflop/cpuRefTime/3.33/6.0,
            gbytes/cpuRefTime, 100.0*gbytes/cpuRefTime/32.0);
    printf("MKL-mic  % 1.8f  % 2.8f  %02.f   %02.8f   %02.f\n", micRefTime,
            gflop/micRefTime, 100.0*gflop/micRefTime/1.053/60.0,
            gbytes/micRefTime, 100.0*gbytes/micRefTime/320.0);

    return 0;
}
