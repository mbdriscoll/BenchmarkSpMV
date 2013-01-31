#define THREADS_PER_BLOCK 128

__global__ void
spmv(int m, int nnz, const int* M_rows, const int* M_cols, const float* M_vals, const float* V_in, float* V_out)
{
    int row = blockIdx.x * blockDim.x;
    if (row >= m)
        return;

    int c = threadIdx.x;

    __shared__ float tmp[THREADS_PER_BLOCK];

    int lb = M_rows[row]-1,
        ub = M_rows[row+1]-1;

    if (c < ub-lb)
        tmp[c] = M_vals[lb+c] * V_in[ M_cols[lb+c]-1 ];
    else
        tmp[c] = 0.0;

    __syncthreads();

    /* This uses a tree structure to do the addtions */
    for (int stride = blockDim.x/2; stride >  0; stride /= 2) {
        if (c < stride)
            tmp[c] += tmp[c + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        V_out[row] = tmp[0];
}

extern "C" {

#include "driver.h"
#include <cusparse_v2.h>

cusparseStatus_t
my_cusparseScsrmv(cusparseHandle_t handle, cusparseOperation_t transA,
    int m, int n, int nnz, float* alpha,
    cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA, const int *csrColIndA,
    const float *x, float* beta,
    float *y ) {

    const int* M_rows = csrRowPtrA;
    const int* M_cols = csrColIndA;
    const float* M_vals = csrValA;
    const float* V_in = x;
    float* V_out = y;

    int blks = (m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    spmv<<<blks,THREADS_PER_BLOCK>>>(m, nnz, M_rows, M_cols, M_vals, V_in, V_out);

    return CUSPARSE_STATUS_SUCCESS;
}

} // extern C
