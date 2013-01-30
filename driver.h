#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cfloat>

#include <sys/time.h>

extern "C" {
#include "mmio.h"
#include <mkl_spblas.h>
}

#include <cuda_runtime.h>
#include <cusparse_v2.h>

class CsrMatrix {
  public:
    cusparseMatDescr_t desc;
    int m, n, nnz;
    int *rows;
    int *cols;
    float *vals;

    CsrMatrix(int m, int n, int nnz, int* rows, int *cols, float *vals) :
        m(m), n(n), nnz(nnz), rows(rows), cols(cols), vals(vals)
    { }
};

class HostCsrMatrix : public CsrMatrix {
  public:
    HostCsrMatrix(int m, int n, int nnz, int *rows, int *cols, float *vals) :
        CsrMatrix(m, n, nnz, rows, cols, vals)
    { }
};

class DeviceCsrMatrix : public CsrMatrix {
  public:
    DeviceCsrMatrix(int m, int n, int nnz, int *rows, int *cols, float *vals);
};

