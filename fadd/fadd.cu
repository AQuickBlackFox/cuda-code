#include<iostream>
#include <stdio.h>

#define ITER 1024*1024*16

#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

__global__ void XnorPopCntAccum(float *in1d, float* outd) {
    int tx = threadIdx.x;
    float in1 = in1d[tx];
    float out = outd[tx];
    for (int i = 0; i < ITER; i++) {
      out += in1;
    }
    outd[tx] = out;
}


int main(int argc, char* argv[]) {
    cudaDeviceProp prop;
    unsigned device = 0;
    unsigned WI = 512;
    unsigned CU = 28;
    if(argc == 4){
        device = atoi(argv[1]);
        WI = atoi(argv[2]);
        CU = atoi(argv[3]);
    }
    cudaSetDevice(device);
    cudaGetDeviceProperties(&prop, device);
    std::cout<<prop.name<<std::endl;
    float *in1d, *outd;
    cudaMalloc((void**)&in1d, WI*4);
    cudaMalloc((void**)&outd, WI*4);
    XnorPopCntAccum<<<1, WI>>>(in1d, outd);  // warm up
    cudaDeviceSynchronize();
    unsigned long long dt = dtime_usec(0);
    XnorPopCntAccum<<<CU, WI>>>(in1d, outd);
    cudaDeviceSynchronize();
    dt = dtime_usec(dt);
    unsigned long long ops = ITER;
    ops *= CU;
    ops *= WI;
    float et = dt/(float)USECPSEC;
    unsigned long long Mops = ops/1000000;
    std::cout<<"NumThreads = "<<WI<<" NumBlocks = "<<CU<<" ITER = "<<ITER<<std::endl;
    std::cout<<et<<"s for "<< Mops << " FAdd Mops"<<std::endl;
    float tp = (Mops)/(et*1000000);
    std::cout << "throughput: " << tp << " Tops/s" << std::endl;
}
