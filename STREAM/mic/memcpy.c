#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>

#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE (1<<30)
#endif

int main() {
    char *arr_src = (char*) malloc(STREAM_ARRAY_SIZE),
         *arr_dst = (char*) malloc(STREAM_ARRAY_SIZE);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    {
#if 0
        memcpy(arr_dst, arr_src, STREAM_ARRAY_SIZE);
#else
        #pragma omp parallel for
        for(int i = 0; i < STREAM_ARRAY_SIZE; i++)
            arr_dst[i] = arr_src[i];
#endif
    }
    gettimeofday(&end, NULL);

    // precision check
    double sec = (end.tv_sec  - start.tv_sec),
           usec = (end.tv_usec - start.tv_usec);
    if (sec == 0.0 && usec < 20.0)
        printf("WARNING: not enough sample time: %d ticks.\n", (int) usec);

    // print results
    double GB = 1.e-9 * 2.0 * (double) STREAM_ARRAY_SIZE;
    double gbps = GB / (sec + 1e-6*usec);
    printf("Saw bandwidth: %f GB/s (%f usec, %f GB)\n", gbps, usec, GB);
    printf("Last elem: %c\n", arr_dst[STREAM_ARRAY_SIZE-1]);

    return 0;
}
