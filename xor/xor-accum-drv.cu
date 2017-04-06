#include <iostream>
#include <stdio.h>
#include <cuda.h>
#define ITER 1024*1024*16
#define WI 512

#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

#define fileName "xor-accum.ptx"
#define kernelName "XorAccum"

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

int main(int argc, char* argv[]) {
    unsigned deviceId = 0;
    if(argc == 2){
        deviceId = atoi(argv[1]);
    }
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUdeviceptr in1d, outd;
    CUresult err = cuInit(0);
    char name[100];
    cuDeviceGet(&device, deviceId);
    cuDeviceGetName(name, 100, device);
    std::cout<<name<<std::endl;
    int CU;
    cuDeviceGetAttribute(&CU, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    CU = CU *4;
    cuCtxCreate(&context, 0, deviceId);
    cuModuleLoad(&module, fileName);
    cuModuleGetFunction(&function, module, kernelName);
    cuMemAlloc(&in1d, WI*4);
    cuMemAlloc(&outd, WI*4);
    void *args[2] = {&in1d, &outd};
    cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);  // warm-up
    cudaDeviceSynchronize();
    unsigned long long dt = dtime_usec(0);
    cuLaunchKernel(function, CU, 1, 1, WI, 1, 1, 0, 0, args, 0);
    cudaDeviceSynchronize();
    dt = dtime_usec(dt);
    unsigned long long ops = ITER;
    ops *= CU;
    ops *= WI;
    float et = dt/(float)USECPSEC;
    unsigned long long Mops = ops/1000000;
    std::cout<<et<<"s for "<< Mops << " XNOR+ACCUM Mops"<<std::endl;
    float tp = (Mops)/(et*1000000);
    std::cout << "throughput: " << tp << " Tops/s" << std::endl;
}
