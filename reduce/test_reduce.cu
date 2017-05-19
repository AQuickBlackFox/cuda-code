#include<cuda.h>
#include"cuda_runtime_api.h"
#include<iostream>

/*
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

using namespace cub;
*/

template<typename T, int NUM_THREADS>
class ReductionBlock{
    T sum = 0;
public:
    __device__ void Load(T* data, int n) {
        unsigned i = blockIdx.x * NUM_THREADS * 2 + threadIdx.x;
        unsigned gridSize = NUM_THREADS * 2 * gridDim.x;
        while(i < n) {
            sum += data[i];
            if(i + NUM_THREADS < n) {
                sum += data[i+NUM_THREADS];
            }
            i += gridSize;
        }
    }
    __device__ T Sum(){
        __shared__ T sdata[NUM_THREADS];
        int tid = threadIdx.x;
        sdata[tid] = sum;
        __syncthreads();
        if((NUM_THREADS >= 512) && (tid < 256)) {
            sum = sum + sdata[tid + 256];
            sdata[tid] = sum;
        }
        __syncthreads();
        if((NUM_THREADS >= 256) && (tid < 128)) {
            sum = sum + sdata[tid + 128];
            sdata[tid] = sum;
        }
        __syncthreads();
        if((NUM_THREADS >= 128) && (tid < 64)) {
            sum = sum + sdata[tid + 64];
            sdata[tid] = sum;
        }
        __syncthreads();
        if((NUM_THREADS >= 64) && (tid < 32)) {
            sum = sum + sdata[tid + 32];
            sdata[tid] = sum;
        }
        __syncthreads();
        if((NUM_THREADS >= 32) && (tid < 16)) {
            sum = sum + sdata[tid + 16];
            sdata[tid] = sum;
        }
        __syncthreads();
        if((NUM_THREADS >= 16) && (tid < 8)) {
            sum = sum + sdata[tid + 8];
            sdata[tid] = sum;
        }
        __syncthreads();
        if((NUM_THREADS >= 8) && (tid < 4)) {
            sum = sum + sdata[tid + 4];
            sdata[tid] = sum;
        }
        __syncthreads();
        if((NUM_THREADS >= 4) && (tid < 2)) {
            sum = sum + sdata[tid + 2];
            sdata[tid] = sum;
        }
        __syncthreads();
        if((NUM_THREADS >= 2) && (tid < 1)) {
            sum = sum + sdata[tid + 1];
            sdata[tid] = sum;
        }
        __syncthreads;
        return sum;
    }

};

/*
template<
    int BLOCK_THREADS,
    int ITEMS_PER_THREAD,
    BlockReduceAlgorithm ALGORITHM>
__global__ void BlockSumKernel(
        int *d_in,
        int *d_out,
        clock_t *d_elapsed)
{
    typedef BlockReduce<int, BLOCK_THREADS, ALGORITHM> BlockReduceT;

    __shared__ typename BlockReduceT::TempStorage temp_storage;

    int data[ITEMS_PER_THREAD];
    LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_in, data);

    clock_t start = clock();

    int aggregate = BlockReduceT(temp_storage).Sum(data);

    clock_t stop = clock();

    if(threadIdx.x == 0)
    {
        *d_elapsed = (start > stop) ? start - stop : stop - start;
        *d_out = aggregate;
    }
}
*/

template<int NUM_THREADS>
__global__ void BlockSumKernel(
    int *d_in,
    int *d_out,
    int n)
{
    ReductionBlock<int, NUM_THREADS> T;
    T.Load(d_in, n);
    int val = T.Sum();
    if(threadIdx.x == 0) d_out[blockIdx.x] = val;
}

int Initialize(int *h_in, int *h_out, int num_items)
{
    int inclusive = 0;
    for(int i=0;i<num_items;++i)
    {
        h_in[i] = 10;
        h_out[i] = 0;
        inclusive += h_in[i];
    }
}

int main()
{
    const int BLOCK_THREADS = 128;
    const int ITEMS_PER_THREAD = 1;

    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int *h_in = new int[TILE_SIZE];
    int *h_gpu = new int[TILE_SIZE];
    int h_aggregate = Initialize(h_in, h_gpu, TILE_SIZE);

    int *d_in, *d_out;
    cudaMalloc(&d_in, TILE_SIZE * sizeof(int));
    cudaMalloc(&d_out, TILE_SIZE * sizeof(int));

    cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_gpu, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice);
//    BlockSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_REDUCE_WARP_REDUCTIONS> <<<dim3(1,1,1), dim3(128,1,1) >>>(d_in, d_out, d_elapsed);

    BlockSumKernel<BLOCK_THREADS> <<<dim3(1,1,1), dim3(128,1,1) >>>(d_in, d_out, BLOCK_THREADS);
    cudaMemcpy(h_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout<<h_gpu[0]<<std::endl;
}
