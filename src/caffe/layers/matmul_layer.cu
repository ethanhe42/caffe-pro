#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/matmul_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MatmulLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* A = bottom[0]->gpu_data();
  const Dtype* B = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  caffe_copy(count, eye_.gpu_data(), top_data);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          A, B, (Dtype)(-1.), top_data);
}

template <typename Dtype>
void MatmulLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
        
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* A = bottom[0]->gpu_data();      
    const Dtype* B = bottom[1]->gpu_data();      
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        M_, K_, N_,
        (Dtype)1., top_diff, B, //TODO: should * 2
        (Dtype)0., bottom[0]->mutable_gpu_diff());
  }
  if (propagate_down[1]) {
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
        N_,K_, M_,
        (Dtype)1., top_diff, A, //TODO: should * 2
        (Dtype)0., bottom[1]->mutable_gpu_diff());
  }        
}

INSTANTIATE_LAYER_GPU_FUNCS(MatmulLayer);

}  // namespace caffe
