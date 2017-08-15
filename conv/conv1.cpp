#include<iostream>
#include<vector>
#include<cassert>
#include"cuda_runtime.h"

typedef float U;

struct Image {
    U *hPtr, *dPtr;
    size_t h, w;
};

struct Filter {
    U *hPtr, *dPtr;
    size_t h, w;
};

void hostConvolution(std::vector<Image> &outputs, std::vector<Image> &inputs, Filter &filter) {
    size_t h = inputs[0].h;
    size_t w = inputs[0].w;
    for(int image_id = 0; image_id < outputs.size(); image_id++) {
        for(int j=1; j<h-1; j++) {
            for(int i=1; i<w-1; i++) {
                outputs[image_id].hPtr[i+j*w] = \ 
                    filter.hPtr[0] * inputs[image_id].hPtr[(i-1) + (j-1)*w] + \
                    filter.hPtr[1] * inputs[image_id].hPtr[(i+0) + (j-1)*w] + \
                    filter.hPtr[2] * inputs[image_id].hPtr[(i+1) + (j-1)*w] + \
                    filter.hPtr[3] * inputs[image_id].hPtr[(i-1) + (j+0)*w] + \
                    filter.hPtr[4] * inputs[image_id].hPtr[(i+0) + (j+0)*w] + \
                    filter.hPtr[5] * inputs[image_id].hPtr[(i+1) + (j+0)*w] + \
                    filter.hPtr[6] * inputs[image_id].hPtr[(i-1) + (j+1)*w] + \
                    filter.hPtr[7] * inputs[image_id].hPtr[(i+0) + (j+1)*w] + \
                    filter.hPtr[8] * inputs[image_id].hPtr[(i+1) + (j+1)*w];
            }
        }
    }
}

int main() {
    const size_t filter_h = 3;
    const size_t filter_w = 3;
    const size_t image_h = 256;
    const size_t image_w = 256;
    const size_t num_images = 16;

    std::vector<Image> inputImages(num_images), outputImages(num_images);
    for(int j=0;j<num_images;j++) {
        inputImages[j].hPtr = new U[image_w*image_h];
        inputImages[j].h = image_h;
        inputImages[j].w = image_w;
        cudaMalloc(&inputImages[j].dPtr, image_w*image_h*sizeof(U));
        outputImages[j].hPtr = new U[image_w*image_h];
        outputImages[j].h = image_h;
        outputImages[j].w = image_w;
        cudaMalloc(&outputImages[j].dPtr, image_w*image_h*sizeof(U));
    }

    for(int j=0;j<num_images;j++) {
        for(int i=0;i<image_h * image_w; i++) {
            inputImages[j].hPtr[i] = U(2.0f);
        }
        for(int i=0;i<image_h * image_w; i++) {
            outputImages[j].hPtr[i] = U(1.0f);
        }
    }

    Filter weights;
    weights.hPtr = new U[filter_h*filter_w];
    weights.h = filter_h;
    weights.w = filter_w;

    for(int i=0;i<filter_h*filter_w;i++) {
        weights.hPtr[i] = 1.0f;
    }

    hostConvolution(outputImages, inputImages, weights);

    for(int j=1;j<image_h-1;j++) {
        for(int i=1;i<image_w-1;i++) {
            std::cout<<outputImages[1].hPtr[i+j*image_w]<<" ";
        }
    }
    std::cout<<std::endl;
}
