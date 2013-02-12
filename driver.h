#ifndef _MBD_DRIVER_H_
#define _MBD_DRIVER_H_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cfloat>

#include <sys/time.h>

extern "C" {
#include "mmio.h"
#include <mkl_spblas.h>
}

#define NITER 1000

class CsrMatrix {
  public:
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

class DeviceHybMatrix : public CsrMatrix {
  public:
    DeviceHybMatrix(DeviceCsrMatrix *Mcsr);
};

void check_vec(int n, float *expected, float *actual);

#include "mic.h"

extern float *answer;

#endif /* _MBD_DRIVER_H_ */
