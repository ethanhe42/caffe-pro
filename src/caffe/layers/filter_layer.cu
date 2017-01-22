#include <vector>

#include "caffe/layers/filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void filter_gpu(const int nthreads, const Dtype* from_data,
  Dtype* to_data, const int from, const int to, const int hw, const int chw) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int from_idx = (index / hw ) * chw + from * hw + index % hw;
    int to_idx   = (index / hw ) * chw + to   * hw + index % hw;

    *(to_data + to_idx) = *(from_data + from_idx);
  }
}

template <typename Dtype>
__global__ void filter_zero_gpu(const int nthreads,
  Dtype* to_data, const int to, const int hw, const int chw) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int to_idx   = (index / hw ) * chw + to   * hw + index % hw;
    *(to_data + to_idx) = Dtype(0);
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int new_tops_num = indices_to_forward_.size();
  // forward all filtered items for all bottoms but the Selector (bottom[last])
  for (int t = 0; t < top.size(); ++t) {
    const Dtype* bottom_data = bottom[t]->gpu_data();
    Dtype* top_data = top[t]->mutable_gpu_data();
    int dim = bottom[t]->count() / bottom[t]->shape(axis_);
    for (int n = 0; n < new_tops_num; ++n) {
      int data_offset_top = n * dim;
      int data_offset_bottom = indices_to_forward_[n] * dim;
      if (axis_) {
            filter_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, bottom[t]->gpu_data(), top[t]->mutable_gpu_data(), indices_to_forward_[n], n, bottom[t]->count(1), bottom[t]->count(2));
      }
      else {
        caffe_copy(dim, bottom_data + data_offset_bottom,
          top_data + data_offset_top);
      }
    }
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[bottom.size() - 1]) {
    LOG(FATAL) << this->type()
               << "Layer cannot backpropagate to filter index inputs";
  }
  for (int i = 0; i < top.size(); ++i) {
    // bottom[last] is the selector and never needs backpropagation
    // so we can iterate over top vector because top.size() == bottom.size() -1
    if (propagate_down[i]) {
      const int dim = top[i]->count() / top[i]->shape(axis_);
      const int hw = top[i]->count(2);
      const int chw = top[i]->count(1);
      int next_to_backward_offset = 0;
      int batch_offset = 0;
      int data_offset_bottom = 0;
      int data_offset_top = 0;
      for (int n = 0; n < bottom[i]->shape(0); ++n) {
        if (next_to_backward_offset >= indices_to_forward_.size()) {
          // we already visited all items that were been forwarded, so
          // just set to zero remaining ones
          data_offset_bottom = n * dim;
          if (axis_){
            filter_zero_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, bottom[i]->mutable_gpu_diff(), n, hw, chw);
          } else{
            caffe_gpu_set(dim, Dtype(0),
              bottom[i]->mutable_gpu_diff() + data_offset_bottom);
          }
        } else {
          batch_offset = indices_to_forward_[next_to_backward_offset];
          data_offset_bottom = n * dim;
          if (n != batch_offset) {  // this data was not been forwarded
            if (axis_){
              filter_zero_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
              <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
              dim, bottom[i]->mutable_gpu_diff(), n, hw, chw);
            } else{          
              caffe_gpu_set(dim, Dtype(0),
                  bottom[i]->mutable_gpu_diff() + data_offset_bottom);
            }
          } else {  // this data was been forwarded
            data_offset_top = next_to_backward_offset * dim;
            if (axis_) {
                  filter_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
                  <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
                  dim, top[i]->mutable_gpu_diff(), bottom[i]->mutable_gpu_diff(), next_to_backward_offset, n, hw, chw);
            }
            else {
                  caffe_copy(dim, top[i]->mutable_gpu_diff() + data_offset_top,
                      bottom[i]->mutable_gpu_diff() + data_offset_bottom);
            }            
            ++next_to_backward_offset;  // point to next forwarded item index
          }
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FilterLayer);

}  // namespace caffe
