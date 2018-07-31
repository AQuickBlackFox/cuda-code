//
// 64 * 64 floats
// 2 float2 per workitem
// 16 * 16 workitems per workgroup, each working on 4 floats
// this makes up to 32 * 32 floats
// 4 workgroups gets launched
//

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda.h>
#include "cuda_runtime_api.h"

__global__ void Kernel(float2* matrix_a, float2* matrix_b, float2* matrix_c) {
    unsigned tx = threadIdx.x;
    unsigned ty = threadIdx.y;

    unsigned bx = blockIdx.x;
    unsigned by = blockIdx.y;

    unsigned index_a = tx + bx * 16;
    unsigned index_b = ty + by * 16;

    unsigned index_c0 = tx + bx * 16 + ty * 32 * 2 + by * 2 * 32 * 16;
    unsigned index_c1 = index_c0 + 32;
    float2 c0 = matrix_c[index_c0];
    float2 c1 = matrix_c[index_c1];

    for(size_t k = 0; k < 64; k++) {
    float2 a = matrix_a[index_a + k * 32];
    float2 b = matrix_b[index_b + k * 32];

    c0.x += a.x * b.x;
    c0.y += a.x * b.y;
    c1.x += a.y * b.x;
    c1.y += a.y * b.y;

    }

    matrix_c[index_c0] = c0;
    matrix_c[index_c1] = c1;
}

int main() {
    size_t m = 64, n = 64, k = 64;
    size_t num_iter = 1;
    size_t size = m * n * sizeof(float);
    std::vector<float> A(m * k);
    std::vector<float> B(n * k);
    std::vector<float> C(n * m);

    std::fill(A.begin(), A.end(), 1.0f);
    std::fill(B.begin(), B.end(), 2.0f);
    std::fill(C.begin(), C.end(), 1.0f);;

    float2* Ad, *Bd, *Cd;
    cudaMalloc(&Ad, size);
    cudaMalloc(&Bd, size);
    cudaMalloc(&Cd, size);

    cudaMemcpy(Ad, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(Cd, C.data(), size, cudaMemcpyHostToDevice);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for(size_t i = 0; i < num_iter; i++) {
    Kernel<<<dim3(2,2,1), dim3(16,16,1)>>>(Ad, Bd, Cd);
    }
    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

    std::cout << time << std::endl;

    cudaMemcpy(C.data(), Cd, size, cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < m; j++) {
            std::cout << C[i + j * m] << " ";
        }
        std::cout << std::endl;
    }

}
