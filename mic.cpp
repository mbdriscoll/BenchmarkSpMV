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
