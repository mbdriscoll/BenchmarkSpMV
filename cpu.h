#ifndef _MBD_CPU_H_
#define _MBD_CPU_H

#include "driver.h"

class HostCsrMatrix : public CsrMatrix {
  public:
    HostCsrMatrix(int m, int n, int nnz, int *coo_rows, int *coo_cols, float *coo_vals);
};

double cpuRefSpMV(HostCsrMatrix *M, float *v);

#endif /* _MBD_CPU_H_ */
