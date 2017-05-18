#include <cuda.h>
#include "cuda_runtime_api.h"
#include <iostream>
#include <numeric>

#define LEN 65536
#define SIZE LEN*sizeof(float)

__global__ void reduce0(int *g_idata, int *g_odata)
{
    __shared__ int sdata[1024];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for(unsigned s = 1; s < blockDim.x; s *= 2)
    {
        if(tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main()
{
    int *A = new int[LEN];
    int *B = new int[LEN];

    int *Ad, *Bd;

    for(int i=0;i<LEN;i++) {
        A[i] = 1;
        B[i] = 0;
    }

    cudaMalloc(&Ad, SIZE);
    cudaMalloc(&Bd, SIZE);

    cudaMemcpy(Ad, A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, SIZE, cudaMemcpyHostToDevice);
    dim3 dimGrid(LEN/1024,1,1);
    reduce0<<<dimGrid, dim3(1024,1,1)>>>(Ad, Bd);
    cudaDeviceSynchronize();
    cudaMemcpy(B, Bd, SIZE, cudaMemcpyDeviceToHost);    
    std::cout<<std::accumulate(B, B+dimGrid.x, 0)<<std::endl;
}
