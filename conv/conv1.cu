#include<iostream>
#include<cuda.h>
#include"cuda_runtime.h"

#define UNROLL 4
#define WIDTH 256
#define FIL_X 3
#define FIL_Y 3

template<typename T, typename U, typename V>
__global__ void Conv1(T *Output, T *Input, T *Filter, size_t h, size_t w, size_t f_h, size_t f_w){
  int tx = threadIdx.x;
  __shared__ T sIn[4][WIDTH];
  T f[FIL_Y][FIL_X], x[FIL_X], y;

  for(int j=0;j<FIL_Y;j++) {
    for(int i=0;i<FIL_X;i++){
      f[j][i] = Filter[i+j*FIL_X];
    }
  }

  for(int i=0;i<4;i++){
    sIn[i][tx] = Input[i*w+tx];
  }

  x[0] = sIn[0][tx-1 > 0 ? tx : 0];
  x[1] = sIn[0][tx];
  x[2] = sIn[0][tx+1 < 255 ? tx : 255];

  y  = x[0] * f[0][0];
  y += x[1] * f[0][1];
  y += x[2] * f[0][2];

  x[0] = sIn[1][tx-1 > 0 ? tx : 0];
  x[1] = sIn[1][tx];
  x[2] = sIn[1][tx+1 < 255 ? tx : 255];

  y += x[0] * f[1][0];
  y += x[1] * f[1][1];
  y += x[2] * f[1][2];

  x[0] = sIn[0][tx-1 > 0 ? tx : 0];
  x[1] = sIn[0][tx];
  x[2] = sIn[0][tx+1 < 255 ? tx : 255];

  y += x[0] * f[2][0];
  y += x[1] * f[2][1];
  y += x[2] * f[2][2];

  Output[tx] = y;
}

int main() {
  float *In, *Out, *Fil;
  In = new float[UNROLL*WIDTH];
  Out = new float[UNROLL*WIDTH];
  Fil = new float[FIL_X*FIL_Y];

  for(int i=0;i<UNROLL*WIDTH;i++){
    In[i] = 2.0f;
    Out[i] = 0.0f;
  }

  for(int i=0;i<FIL_X*FIL_Y;i++){
    Fil[i] = 1.0f;
  }

  float *Ind, *Outd, *Fild;
  cudaMalloc(&Ind, sizeof(float)*UNROLL*WIDTH);
  cudaMalloc(&Outd, sizeof(float)*UNROLL*WIDTH);
  cudaMalloc(&Fild, sizeof(float)*FIL_X*FIL_Y);

  cudaMemcpy(Ind, In, sizeof(float)*UNROLL*WIDTH, cudaMemcpyHostToDevice);
  cudaMemcpy(Outd, Out, sizeof(float)*UNROLL*WIDTH, cudaMemcpyHostToDevice);
  cudaMemcpy(Fild, Fil, sizeof(float)*FIL_X*FIL_Y, cudaMemcpyHostToDevice);

  Conv1<float, float, float><<<dim3(1,1,1), dim3(WIDTH,1,1)>>>(Outd, Ind, Fild, UNROLL, WIDTH, 3, 3);
  cudaDeviceSynchronize();

  cudaMemcpy(Out, Outd, sizeof(float)*UNROLL*WIDTH, cudaMemcpyDeviceToHost);

  std::cout<<Out[10]<<std::endl;
}

