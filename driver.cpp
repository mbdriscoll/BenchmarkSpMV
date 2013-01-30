#include "driver.h"

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
        0, // job(2)=1 (zero-based indexing for csr matrix)
        1, // job(3)=1 (one-based indexing for coo matrix)
        0, // empty
        nnz, // job(5)=nnz (sets nnz for csr matrix)
        0  // job(6)=0 (all output arrays filled)
    };

    int info;
    mkl_scsrcoo(job, &m, csr_vals, csr_cols, csr_rows, &nnz, coo_vals, coo_rows, coo_cols, &info);

    *hM = new HostCsrMatrix(m, n, nnz, csr_rows, csr_cols, csr_vals);
    *dM = new DeviceCsrMatrix(m, n, nnz, csr_rows, csr_cols, csr_vals);

    free(coo_rows);
    free(coo_cols);
    free(coo_vals);
    /* don't free csr_rows etc. keep for hM to use */
}

DeviceCsrMatrix::DeviceCsrMatrix(int m, int n, int nnz, int *h_rows, int *h_cols, float *h_vals) :
    CsrMatrix(m, n, nnz, NULL, NULL, NULL) {
        cudaMalloc(&rows, (m+1) * sizeof(int));
        cudaMalloc(&cols, nnz * sizeof(int));
        cudaMalloc(&vals, nnz * sizeof(float));

        cudaMemcpy(rows, h_rows, (m+1)*sizeof(int),  cudaMemcpyHostToDevice);
        cudaMemcpy(cols, h_cols, nnz*sizeof(int),  cudaMemcpyHostToDevice);
        cudaMemcpy(vals, h_vals, nnz*sizeof(float),  cudaMemcpyHostToDevice);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s MATRIX.mm\n", argv[0]);
        exit(1);
    }

    printf("Reading matrix at %s.\n", argv[1]);
    HostCsrMatrix *hM;
    DeviceCsrMatrix *dM;
    mm_read(argv[1], &hM, &dM);

    printf("Benchmarking.\n");

    printf("Done.\n");
    return 0;
}
