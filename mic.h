#ifndef _MBD_MIC_H_
#define _MBD_MIC_H

#include "driver.h"

double micRefSpMV(HostCsrMatrix *M, float *v);

class DeviceCsrMatrix : public CsrMatrix {
  public:
    DeviceCsrMatrix(int m, int n, int nnz, int *coo_rows, int *coo_cols, float *coo_vals);
};

class DeviceHybMatrix : public CsrMatrix {
  public:
    DeviceHybMatrix(DeviceCsrMatrix *Mcsr);
};

#endif /* _MBD_MIC_H */
