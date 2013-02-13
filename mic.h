#ifndef _MBD_MIC_H_
#define _MBD_MIC_H

#include "driver.h"

class DeviceCsrMatrix : public CsrMatrix {
  public:
    DeviceCsrMatrix(int m, int n, int nnz, int *coo_rows, int *coo_cols, float *coo_vals);
    ~DeviceCsrMatrix();
};

double micRefSpMV(DeviceCsrMatrix *M, float *v);

#endif /* _MBD_MIC_H */
