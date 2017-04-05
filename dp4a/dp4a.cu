#include<iostream>
#include <stdio.h>

#define ITER 1024*1024
#define SSZ 512
#define BSZ 7*4


#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

__global__ void DoDP4A(int *in1d, int *in2d, int* in3d, int* outd) {
    int tx = threadIdx.x;
    int in1 = in1d[tx];
    int in2 = in2d[tx];
    int in3 = in3d[tx];
    int out;
    for (int i = 0; i < ITER; i++) {
      out += __dp4a(in1, in2, in3);
    }
    outd[tx] = out;
}


int main() {
    cudaSetDevice(2);
    int *in1d, *in2d, *in3d, *outd;
    cudaMalloc((void**)&in1d, SSZ*4);
    cudaMalloc((void**)&in2d, SSZ*4);
    cudaMalloc((void**)&in3d, SSZ*4);
    cudaMalloc((void**)&outd, SSZ*4);
    DoDP4A<<<1, SSZ>>>(in1d, in2d, in3d, outd);  // warm up
    cudaDeviceSynchronize();
    unsigned long long dt = dtime_usec(0);
    DoDP4A<<<BSZ, SSZ>>>(in1d, in2d, in3d, outd);
    cudaDeviceSynchronize();
    dt = dtime_usec(dt);
    unsigned long long ops = ITER;
    ops *= BSZ;
    ops *= SSZ;
    float et = dt/(float)USECPSEC;
    unsigned long long Mops = ops/1000000;
    std::cout<<et<<"s for "<< Mops << " Mdp4a"<<std::endl;
    float tp = (Mops*8)/(et*1000000);
    std::cout << "throughput: " << tp << " Tops/s" << std::endl;
}
