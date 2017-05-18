#include <cuda.h>
#include "cuda_runtime_api.h"
#include <iostream>
#include <numeric>

#define LEN 1024*16
#define SIZE LEN*sizeof(int)

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


__global__ void reduce1(int *g_idata, int *g_odata)
{
    __shared__ int sdata[1024];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for(unsigned s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s *tid;
        if(index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce2(int *g_idata, int *g_odata)
{
    __shared__ int sdata[1024];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for(unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce3(int *g_idata, int *g_odata)
{
    __shared__ int sdata[1024];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * (blockDim.x * 2)+ threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    for(unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce5(int *g_idata, int *g_odata, const unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x* blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    int mySum = 0;

    while (i < n)
    {
        mySum += g_idata[i];
        if (i + blockSize < n)
            mySum += g_idata[i+blockSize];
        i += gridSize;
    }

    sdata[tid] = mySum;
    __syncthreads();


    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        volatile int* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

/*
This piece of kernel is borrowed from CUDA toolkit
*/

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    extern __shared__ T sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if 1
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}


void doReduce0()
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
//    reduce0<<<dim3(1,1,1), dim3(dimGrid.x,1,1)>>>(Bd, Bd);
    cudaDeviceSynchronize();
    cudaMemcpy(B, Bd, SIZE, cudaMemcpyDeviceToHost);    
    std::cout<<B[0]<<std::endl;
    delete A;
    delete B;
    cudaFree(Ad);
    cudaFree(Bd);
}


void doReduce1()
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
    reduce1<<<dimGrid, dim3(1024,1,1)>>>(Ad, Bd);
//    reduce1<<<dim3(1,1,1), dim3(dimGrid.x,1,1)>>>(Bd, Bd);
    cudaDeviceSynchronize();
    cudaMemcpy(B, Bd, SIZE, cudaMemcpyDeviceToHost);    
    std::cout<<B[0]<<std::endl;
    delete A;
    delete B;
    cudaFree(Ad);
    cudaFree(Bd);
}


void doReduce2()
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
    reduce2<<<dimGrid, dim3(1024,1,1)>>>(Ad, Bd);
//    reduce2<<<dim3(1,1,1), dim3(dimGrid.x,1,1)>>>(Bd, Bd);
    cudaDeviceSynchronize();
    cudaMemcpy(B, Bd, SIZE, cudaMemcpyDeviceToHost);    
    std::cout<<B[0]<<std::endl;
    delete A;
    delete B;
    cudaFree(Ad);
    cudaFree(Bd);
}

void doReduce3()
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
    reduce3<<<dimGrid, dim3(1024,1,1)>>>(Ad, Bd);
//    reduce3<<<dim3(1,1,1), dim3(dimGrid.x,1,1)>>>(Bd, Bd);
    cudaDeviceSynchronize();
    cudaMemcpy(B, Bd, SIZE, cudaMemcpyDeviceToHost);    
    std::cout<<B[0]<<std::endl;
    delete A;
    delete B;
    cudaFree(Ad);
    cudaFree(Bd);
}


void doReduce6()
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
    dim3 dimBlock(1024,1,1);
    
    reduce6<int, 1024, true> <<<dimGrid, dim3(1024,1,1), dimBlock.x*2*sizeof(int)>>>(Ad, Bd, LEN);
    cudaDeviceSynchronize();
    cudaMemcpy(B, Bd, SIZE, cudaMemcpyDeviceToHost);    
    std::cout<<B[0]<<std::endl;
    delete A;
    delete B;
    cudaFree(Ad);
    cudaFree(Bd);
}





int main(){
    doReduce0();
    doReduce1();
    doReduce2();
    doReduce3();
    doReduce6();
}
