/**
* Calculate Kullback-Leibler Divergence using Thrust
* This sample calcualte KL-Divergence between
* Uniform Normal Distribution and Bernoulli Distribution
* val = - reduce( p[xi] * log(p[xi]/q[xi]) )
*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cstdlib>

struct entropy {
    __device__
    float operator() (const float& x, const float& y) const {
        return x * (log2f(x/y)); //log2
    }

};

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> urd(1.0, 2.0);
    std::bernoulli_distribution bd(0.25);
    thrust::host_vector<float> h_vec1(32 << 10);
    thrust::host_vector<float> h_vec2(32 << 10);
    std::fill(h_vec1.begin(), h_vec1.end(), urd(gen));
    std::fill(h_vec2.begin(), h_vec2.end(), bd(gen));
    thrust::device_vector<float> d_vec1 = h_vec1;
    thrust::device_vector<float> d_vec2 = h_vec2;
    thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_vec2.begin(), entropy());
    float val = -thrust::reduce(d_vec2.begin(), d_vec2.end());
    std::cout<<val<<std::endl;
    return 0;
}

