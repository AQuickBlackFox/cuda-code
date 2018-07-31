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

__global__ void Kernel(float4* matrix_a, float4* matrix_b, float4* matrix_c) {
    unsigned tx = threadIdx.x;
    unsigned ty = threadIdx.y;
    unsigned bx = blockIdx.x;
    unsigned by = blockIdx.y;

    unsigned bxmod2 = bx % 2;
    unsigned bxdiv2 = bx / 2;

    unsigned index_a = tx + bxmod2 * 64 * 32 + bxdiv2 * 16;
    unsigned index_b = ty + by * 16 + bxmod2 * 64 * 32;

    unsigned index_c0 = tx + bxdiv2 * 16 + ty * (128 / 4) * 4 + by * (128 / 4) * 64;
    unsigned index_c1 = index_c0 + (128/4);
    unsigned index_c2 = index_c1 + (128/4);
    unsigned index_c3 = index_c2 + (128/4);

    float4 c0 = {0.0f}, c1 = c0, c2 = c0, c3 = c0;;
    

    for(size_t i = 0; i < 64; i++) {
        float4 a = matrix_a[index_a + i * 32];
        float4 b = matrix_b[index_b + i * 32];

        c0.x += a.x * b.x;
        c0.y += a.y * b.x;
        c0.z += a.z * b.x;
        c0.w += a.w * b.x;

        c1.x += a.x * b.y;
        c1.y += a.y * b.y;
        c1.z += a.z * b.y;
        c1.w += a.w * b.y;

        c2.x += a.x * b.z;
        c2.y += a.y * b.z;
        c2.z += a.z * b.z;
        c2.w += a.w * b.z;

        c3.x += a.x * b.w;
        c3.y += a.y * b.w;
        c3.z += a.z * b.w;
        c3.w += a.w * b.w;
    }
    float* ptr = reinterpret_cast<float*>(matrix_c + index_c0);
    atomicAdd(ptr, c0.x);
    ptr++;
    atomicAdd(ptr, c0.y);
    ptr++;
    atomicAdd(ptr, c0.z);
    ptr++;
    atomicAdd(ptr, c0.w);

    ptr = reinterpret_cast<float*>(matrix_c + index_c1);
    atomicAdd(ptr, c1.x);
    ptr++;
    atomicAdd(ptr, c1.y);
    ptr++;
    atomicAdd(ptr, c1.z);
    ptr++;
    atomicAdd(ptr, c1.w);

    ptr = reinterpret_cast<float*>(matrix_c + index_c2);
    atomicAdd(ptr, c2.x);
    ptr++;
    atomicAdd(ptr, c2.y);
    ptr++;
    atomicAdd(ptr, c2.z);
    ptr++;
    atomicAdd(ptr, c2.w);

    ptr = reinterpret_cast<float*>(matrix_c + index_c3);
    atomicAdd(ptr, c3.x);
    ptr++;
    atomicAdd(ptr, c3.y);
    ptr++;
    atomicAdd(ptr, c3.z);
    ptr++;
    atomicAdd(ptr, c3.w);
/*
    matrix_c[index_c0] = c0;
    matrix_c[index_c1] = c1;
    matrix_c[index_c2] = c2;
    matrix_c[index_c3] = c3;
*/
}

int main() {
    size_t m = 128, n = 128, k = 128;
    size_t num_iter = 1024;
    size_t size = m * n * sizeof(float);
    std::vector<float> A(m * k);
    std::vector<float> B(n * k);
    std::vector<float> C(n * m);

    std::fill(A.begin(), A.end(), 1.0f);
    std::fill(B.begin(), B.end(), 2.0f);
    std::fill(C.begin(), C.end(), 1.0f);;

    float4* Ad, *Bd, *Cd;
    cudaMalloc(&Ad, size);
    cudaMalloc(&Bd, size);
    cudaMalloc(&Cd, size);

    cudaMemcpy(Ad, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(Cd, C.data(), size, cudaMemcpyHostToDevice);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for(size_t i = 0; i < num_iter; i++) {
    Kernel<<<dim3(4,2,1), dim3(16,16,1)>>>(Ad, Bd, Cd);
    }
    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

    std::cout << time << std::endl;

    cudaMemcpy(C.data(), Cd, size, cudaMemcpyDeviceToHost);
/*
    for(size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < m; j++) {
            std::cout << C[i + j * m] << " ";
        }
        std::cout << std::endl;
    }
*/
}
