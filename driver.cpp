#include "driver.h"

extern "C" {
#include "mmio.h"
}

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

    printf("reading matrix file\n");
    for (int i = 0; i < nnz; i++)
        status = fscanf(mmfile, "%d %d %g\n", &coo_rows[i], &coo_cols[i], &coo_vals[i]);

    printf("building host matrix\n");
    *hM = new HostCsrMatrix(m, n, nnz, coo_rows, coo_cols, coo_vals);

    printf("building device matrix\n");
    *dM = new DeviceCsrMatrix(m, n, nnz, coo_rows, coo_cols, coo_vals);

    free(coo_rows);
    free(coo_cols);
    free(coo_vals);

    printf("done building matrices\n");
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

    fprintf(stderr, "Found %d/%d errors in answer.\n", errors, n);
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

    float *v = randvec(hM->n);

    printf("running mkl-host tests\n");
    double cpuRefTime = cpuRefSpMV(hM, v);

    printf("running mkl-mic tests\n");
    double micRefTime = micRefSpMV(dM, v);

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
