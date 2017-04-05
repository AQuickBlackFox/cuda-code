#include<iostream>
#include <stdio.h>

#define ITER 1024*1024*16
#define WI 512

#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

__global__ void XorPopCntAccum(int *in1d, int* outd) {
    int tx = threadIdx.x;
    int in1 = in1d[tx];
    int out = outd[tx];
    for (int i = 0; i < ITER; i++) {
      out += __popc((out ^ in1));
    }
    outd[tx] = out;
}


int main() {
    cudaDeviceProp prop;
    unsigned device = 0;
    cudaSetDevice(device);
    cudaGetDeviceProperties(&prop, device);
    std::cout<<prop.name<<std::endl;
    unsigned CU = prop.multiProcessorCount;
    int *in1d, *outd;
    cudaMalloc((void**)&in1d, WI*4);
    cudaMalloc((void**)&outd, WI*4);
    XorPopCntAccum<<<1, WI>>>(in1d, outd);  // warm up
    cudaDeviceSynchronize();
    unsigned long long dt = dtime_usec(0);
    XorPopCntAccum<<<CU, WI>>>(in1d, outd);
    cudaDeviceSynchronize();
    dt = dtime_usec(dt);
    unsigned long long ops = ITER;
    ops *= CU;
    ops *= WI;
    float et = dt/(float)USECPSEC;
    unsigned long long Mops = ops/1000000;
    std::cout<<et<<"s for "<< Mops << " XOR+POPCNT+ACCUM"<<std::endl;
    float tp = (Mops)/(et*1000000);
    std::cout << "throughput: " << tp << " Tops/s" << std::endl;
}
