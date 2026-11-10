// Harness for 17-overlap. Do not edit.

#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

int main(int argc, char** argv) {
    size_t total = (argc > 1) ? size_t(std::atoll(argv[1])) : (size_t(1) << 26);
    size_t chunk = (argc > 2) ? size_t(std::atoll(argv[2])) : (size_t(1) << 21);
    int work = (argc > 3) ? std::atoi(argv[3]) : 64;
    int n_streams = (argc > 4) ? std::atoi(argv[4]) : 4;

    bench::Gate gate("17-overlap");

    // Pinned host memory. Without this, cudaMemcpyAsync is not async and the
    // whole exercise silently does nothing.
    float *h_in, *h_serial, *h_pipe;
    CUDA_CHECK(cudaMallocHost(&h_in, total * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_serial, total * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_pipe, total * sizeof(float)));
    for (size_t i = 0; i < total; ++i) h_in[i] = float(i % 1000) * 0.001f;

    float* d_buf;
    size_t buf_floats = size_t(n_streams) * chunk * 2;
    CUDA_CHECK(cudaMalloc(&d_buf, buf_floats * sizeof(float)));

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; ++i)
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

    run_serial(h_in, h_serial, d_buf, total, chunk, work);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_pipelined(h_in, h_pipe, d_buf, total, chunk, work, streams.data(), n_streams);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> a(h_serial, h_serial + total), b(h_pipe, h_pipe + total);
    gate.correctness(bench::compare(b, a, 1e-6), 1e-6);

    double serial_ms = bench::time_ms(
        [&] { run_serial(h_in, h_serial, d_buf, total, chunk, work); }, 2, 8, false);
    double pipe_ms = bench::time_ms(
        [&] {
            run_pipelined(h_in, h_pipe, d_buf, total, chunk, work, streams.data(),
                          n_streams);
        },
        2, 8, false);

    size_t chunks = (total + chunk - 1) / chunk;
    std::printf("\ntotal          : %zu floats (%.0f MB)\n", total,
                total * 4 / 1e6);
    std::printf("chunk          : %zu floats (%.1f MB), %zu chunks\n", chunk,
                chunk * 4 / 1e6, chunks);
    std::printf("streams        : %d,  work per element: %d FMAs\n", n_streams, work);
    std::printf("serial         : %.3f ms\n", serial_ms);
    std::printf("pipelined      : %.3f ms\n", pipe_ms);
    std::printf("H2D+D2H bytes  : %.0f MB each way\n", total * 4 / 1e6);
    std::printf("\nSweep chunk size and stream count. There is an optimum in both.\n");
    std::printf("Then look at it in nsys: you should SEE the overlap.\n");

    gate.speedup(serial_ms, pipe_ms, 1.6, "serial");

    for (auto s : streams) CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_serial));
    CUDA_CHECK(cudaFreeHost(h_pipe));
    return gate.finish();
}
