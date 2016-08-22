#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/matmul_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MatmulLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//   transpose_ = this->layer_param_.matmul_param().transpose();
  
  M_ = bottom[0]->count(0,1);
  N_ = M_;
  K_ = bottom[0]->count(1,2);
}

template <typename Dtype>
void MatmulLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape(2);
  top_shape[0] = M_;
  top_shape[1] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  // fill the I
  eye_.Reshape(top_shape);
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
    this->layer_param_.matmul_param().weight_filler()));
  weight_filler->Fill(&eye_);
}

template <typename Dtype>
void MatmulLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MatmulLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MatmulLayer);
#endif

INSTANTIATE_CLASS(MatmulLayer);
REGISTER_LAYER_CLASS(Matmul);

}  // namespace caffe
