#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/matmul_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MatmulLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  caffe_gpu_set(count, eye_, top_data);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, bottom_data, (Dtype)(-1.), top_data);
}

template <typename Dtype>
void MatmulLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();      
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        M_, K_, N_,
        (Dtype)2., top_diff, bottom_data,
        (Dtype)0., bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MatmulLayer);

}  // namespace caffe
