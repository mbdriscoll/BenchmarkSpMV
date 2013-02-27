#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

extern "C" {
#include <mkl_spblas.h>
#include "extra/mmio.h"
}

/* number of iterations to yield average performance */
#define NITER 1000

/* peak processor constants */
#define CPU_GFLOPS (3.33*6.0)
#define CPU_STREAM_GBS (23.9)
#define GPU_GFLOPS (1.053*60.0)
#define GPU_STREAM_GBS (129.7)

/* convenience macros */
#define ALLOC alloc_if(1) free_if(0)
#define FREE  alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)
#define TEMP  alloc_if(1) free_if(1)


/**
 * Reads a MatrixMarket file and sets the input pointers
 */
void mm_read(char* filename, int *m, int *n, int *nnz,
             int **rowptrs, int **colinds, float **vals) {
    // open file
    FILE* mmfile = fopen(filename, "r");
    assert(mmfile != NULL && "Read matrix file.");

    // read MatrixMarket header
    int status;
    MM_typecode matcode;
    status = mm_read_banner(mmfile, &matcode);
    assert(status == 0 && "Parsed banner.");
    assert(mm_is_matrix(matcode) &&
           mm_is_sparse(matcode) &&
           mm_is_real(matcode));

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
        status = fscanf(mmfile, "%d %d %g\n",
            &coo_rows[i], &coo_cols[i], &coo_vals[i]);

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
    mkl_scsrcoo(job, m, *vals, *colinds, *rowptrs, nnz,
                coo_vals, coo_rows, coo_cols, &info);
    assert(info == 0 && "Converted COO->CSR");

    // free COO matrix
    free(coo_rows);
    free(coo_cols);
    free(coo_vals);
}


/**
 * Allocates and populates a n-vector with random numbers.
 */
float * randvec(int n) {
    float *v = (float*) malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
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
          *gpu_csr_answer = (float*) malloc(m*sizeof(float)),
          *gpu_hyb_answer = (float*) malloc(m*sizeof(float));

    // -----------------------------------------------------------------------
    // Benchmark MKL performance on CPU

    struct timeval start, end;

    // warm cache
    mkl_scsrgemv((char*)"N", &m, vals, rowptrs, colinds, v, cpu_answer);

    double cpuAvgTimeInSec = 0.0;
    for (int i = 0; i < NITER; i++) {
        gettimeofday (&start, NULL);
        {
            mkl_scsrgemv((char*)"N", &m, vals, rowptrs, colinds, v, cpu_answer);
        }
        gettimeofday (&end, NULL);
        cpuAvgTimeInSec += (end.tv_sec  - start.tv_sec) +
                           (end.tv_usec - start.tv_usec) * 1.e-6;
    }
    cpuAvgTimeInSec /= (double) NITER;

    // -----------------------------------------------------------------------
    // Benchmark cuSPARSE performance on GPU

    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);

    cusparseMatDescr_t desc;
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ONE);

    float *d_v, *d_gpu_csr_answer, *d_gpu_hyb_answer, *d_vals;
    int *d_rowptrs, *d_colinds;
    cudaMalloc(&d_v, n*sizeof(float));
    cudaMalloc(&d_gpu_csr_answer, m*sizeof(float));
    cudaMemcpy(d_v, v, n*sizeof(float), cudaMemcpyHostToDevice);

    cusparseStatus_t status;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float alpha = 1.0,
          beta = 0.0;

    // offload data to device
    cudaMemcpy(d_v,          v,       n*sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowptrs,    rowptrs, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colinds,    colinds, nnz*sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals,       vals,    nnz*sizeof(float), cudaMemcpyHostToDevice);

    // warm cache
    cusparseScsrmv(handle, op, m, n, nnz, &alpha, desc,
            d_vals, d_rowptrs, d_colinds, d_v, &beta, d_gpu_csr_answer);

    // do NITER SpMVs with device-resident CSR matrix
    double gpuCsrAvgTimeInSec = 0.0;
    for (int i = 0; i < NITER; i++) {
        gettimeofday (&start, NULL);
        {
            status = cusparseScsrmv(handle, op, m, n, nnz, &alpha, desc,
                    d_vals, d_rowptrs, d_colinds, d_v, &beta, d_gpu_csr_answer);
            cudaThreadSynchronize();
        }
        gettimeofday (&end, NULL);
        gpuCsrAvgTimeInSec += (end.tv_sec  - start.tv_sec) +
                           (end.tv_usec - start.tv_usec) * 1.e-6;
    }
    gpuCsrAvgTimeInSec /= (double) NITER;

    // make hybrid matrix
    cusparseHybMat_t hyb_matrix;
    cusparseCreateHybMat(&hyb_matrix);
    cusparseScsr2hyb(handle, m, n, desc, d_vals, d_rowptrs, d_colinds, hyb_matrix,
            0, CUSPARSE_HYB_PARTITION_MAX);

    // do NITER SpMVs with device-resident CSR matrix
    double gpuHybAvgTimeInSec = 0.0;
    for (int i = 0; i < NITER; i++) {
        gettimeofday (&start, NULL);
        {
            status = cusparseShybmv(handle, op, &alpha, desc, hyb_matrix,
                    d_v, &beta, d_gpu_hyb_answer);
            cudaThreadSynchronize();
        }
        gettimeofday (&end, NULL);
        gpuHybAvgTimeInSec += (end.tv_sec  - start.tv_sec) +
                           (end.tv_usec - start.tv_usec) * 1.e-6;
    }
    gpuHybAvgTimeInSec /= (double) NITER;

    // copy answers back
    cudaMemcpy(gpu_csr_answer, d_gpu_csr_answer, m*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_hyb_answer, d_gpu_hyb_answer, m*sizeof(float), cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_gpu_csr_answer);
    cudaFree(d_gpu_hyb_answer);
    cudaFree(d_v);
    cudaFree(d_rowptrs);
    cudaFree(d_colinds);
    cudaFree(d_vals);

    // -----------------------------------------------------------------------
    // Verify and display results

    // check the solution
    int csr_errors = 0, hyb_errors = 0;
    for (int i = 0; i < m; i++) {
        if (cpu_answer[i] != gpu_csr_answer[i]) csr_errors += 1;
        if (cpu_answer[i] != gpu_hyb_answer[i]) hyb_errors += 1;
    }
    if (csr_errors != 0) printf("WARNING: found %d/%d errors in CSR solution.\n", csr_errors, m);
    if (hyb_errors != 0) printf("WARNING: found %d/%d errors in HYB solution.\n", hyb_errors, m);

    // calculate total bytes and flops
    double gflop = 1.e-9 * 2.0 * nnz;
    double gbytes = 1.e-9 * (
            nnz * sizeof(float) + // vals
            nnz * sizeof(int) + // cols
            (m+1) * sizeof(int) + // rows
            (n+m) * sizeof(float)); // vectors

    // print performance
    printf("Platform       Time         Gflops/s    %%peak Gbytes/s     %%STREAM\n");
    printf("MKL           % 1.8f  % 2.8f  %02.f    %02.8f   %02.f\n", cpuAvgTimeInSec,
            gflop/cpuAvgTimeInSec, 100.0*gflop/cpuAvgTimeInSec/CPU_GFLOPS,
            gbytes/cpuAvgTimeInSec, 100.0*gbytes/cpuAvgTimeInSec/CPU_STREAM_GBS);
    printf("cuSPARSE-csr  % 1.8f  % 2.8f  %02.f    %02.8f   %02.f\n", gpuCsrAvgTimeInSec,
            gflop/gpuCsrAvgTimeInSec, 100.0*gflop/gpuCsrAvgTimeInSec/GPU_GFLOPS,
            gbytes/gpuCsrAvgTimeInSec, 100.0*gbytes/gpuCsrAvgTimeInSec/GPU_STREAM_GBS);
    printf("cuSPARSE-hyb  % 1.8f  % 2.8f  %02.f    %02.8f   %02.f\n", gpuHybAvgTimeInSec,
            gflop/gpuHybAvgTimeInSec, 100.0*gflop/gpuHybAvgTimeInSec/GPU_GFLOPS,
            gbytes/gpuHybAvgTimeInSec, 100.0*gbytes/gpuHybAvgTimeInSec/GPU_STREAM_GBS);

    // release memory
    free(rowptrs);
    free(colinds);
    free(vals);
    free(gpu_csr_answer);
    free(gpu_hyb_answer);
    free(cpu_answer);
    free(v);

    return 0;
}
