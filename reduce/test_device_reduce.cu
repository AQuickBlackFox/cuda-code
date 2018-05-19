#include<cuda.h>
#include"cuda_runtime_api.h"
#include<iostream>
#include<cassert>

#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start) {
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

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
    return inclusive;
}

void Launcher(int *d_in, int *d_out, int N, dim3 dimGrid){
    int n = 1;
    dim3 dimBlock(1,1,1);

    if(N > 256 && N <= 512) {
        dimBlock.x = 512;
    }
    if(N > 128 && N <= 256) {
        dimBlock.x = 256;
    }
    if(N > 64 && N <= 128) {
        dimBlock.x = 128;
    }
    if(N > 32 && N <= 64) {
        dimBlock.x = 64;
    }
    if(N > 16 && N <= 32) {
        dimBlock.x = 32;
    }
    if(N > 8  && N <= 16) {
        dimBlock.x = 16;
    }
    if(N > 4 && N <= 8) {
        dimBlock.x = 8;
    }
    if(N > 2 && N <= 4) {
        dimBlock.x = 4;
    }
    if(N > 1 && N <= 2){
        dimBlock.x = 2;
    }
    if(N == 1) {
        dimBlock.x = 1;
    }

    switch(dimBlock.x){
        case 512:
            BlockSumKernel<512><<<dimGrid, dimBlock>>>(d_in, d_out, N); break;
        case 256:
            BlockSumKernel<256><<<dimGrid, dimBlock>>>(d_in, d_out, N); break;
        case 128:
            BlockSumKernel<128><<<dimGrid, dimBlock>>>(d_in, d_out, N); break;
        case 64:
            BlockSumKernel<64> <<<dimGrid, dimBlock>>>(d_in, d_out, N); break;
        case 32:
            BlockSumKernel<32> <<<dimGrid, dimBlock>>>(d_in, d_out, N); break;
        case 16:
            BlockSumKernel<16> <<<dimGrid, dimBlock>>>(d_in, d_out, N); break;
        case 8:
            BlockSumKernel<8>  <<<dimGrid, dimBlock>>>(d_in, d_out, N); break;
        case 4:
            BlockSumKernel<4>  <<<dimGrid, dimBlock>>>(d_in, d_out, N); break;
        case 2:
            BlockSumKernel<2>  <<<dimGrid, dimBlock>>>(d_in, d_out, N); break;
        case 1:
            BlockSumKernel<1>  <<<dimGrid, dimBlock>>>(d_in, d_out, N); break;
    }
   
}


void DeviceReduce(int *d_in, int *d_out, int N)
{
    if(N <= 512) {
        Launcher(d_in, d_out, N, dim3(1,1,1));
    }
    if(N > 512) {
        dim3 launch0(N/512 + N % 512 > 1 ? 1 : 0, 1, 1);
        Launcher(d_in, d_out, N, launch0);
        Launcher(d_out, d_out, launch0.x, dim3(1,1,1));
    }
}

int main()
{
    for(int BLOCK_THREADS= 8388608; BLOCK_THREADS > 1; BLOCK_THREADS = BLOCK_THREADS - 5555){
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
    unsigned long long dt = dtime_usec(0);
    DeviceReduce(d_in, d_out, BLOCK_THREADS);
    cudaDeviceSynchronize();
    dt = dtime_usec(dt);
    double et = dt / (double)USECPSEC;
    double bw = (double)(sizeof(int) * TILE_SIZE) / et;
    cudaMemcpy(h_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    if(h_gpu[0] != h_aggregate) {
        std::cout<<"Error at: "<<BLOCK_THREADS<<std::endl;
        std::cout<<"Got: "<<h_gpu[0]<<" Expected: "<<h_aggregate<<std::endl;
        exit(0);
    }
    std::cout<<"Running at N = "<<BLOCK_THREADS<<std::endl;
    std::cout<<"BW: "<<bw<<std::endl;
    cudaFree(d_in);
    cudaFree(d_out);
    delete h_in;
    delete h_gpu;
    } 
}
