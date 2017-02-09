#include <vector>

#include "caffe/layers/filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void filter_gpu(const int nthreads, const Dtype* from_data,
  Dtype* to_data, const int from, const int to, const int hw, const int chw, const int cphw) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int from_idx = (index / hw ) * chw + from * hw + index % hw;
    int to_idx   = (index / hw ) * cphw + to   * hw + index % hw;

    *(to_data + to_idx) = *(from_data + from_idx);
  }
}

template <typename Dtype>
__global__ void filter_zero_gpu(const int nthreads,
  Dtype* to_data, const int to, const int hw, const int chw) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int to_idx   = (index / hw ) * chw + to   * hw + index % hw;
    // *(to_data + to_idx) = Dtype(0);
    memset(to_data + to_idx, 0, sizeof(Dtype));
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (!axis_){
    NOT_IMPLEMENTED;
    return;
  }        
  // forward all filtered items for all bottoms but the Selector (bottom[last])
  for (int t = 0; t < top.size(); ++t) {
    int new_tops_num = top[t]->shape(axis_);
    CHECK_EQ(indices_to_forward_.size(), new_tops_num);    
    const Dtype* bottom_data = bottom[t]->gpu_data();
    Dtype* top_data = top[t]->mutable_gpu_data();
    int dim = bottom[t]->count() / bottom[t]->shape(axis_);
    const int hw = bottom[t]->shape(2) * bottom[t]->shape(3);
    const int chw = bottom[t]->shape(1) * bottom[t]->shape(2) * bottom[t]->shape(3);
    const int cphw = new_tops_num * bottom[t]->shape(2) * bottom[t]->shape(3);

    for (int n = 0; n < new_tops_num; ++n) {
      if (axis_) {
            filter_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, bottom_data, top_data, indices_to_forward_[n], n, hw, chw, cphw);
      }
      else {
        int data_offset_top = n * dim;
        int data_offset_bottom = indices_to_forward_[n] * dim;
        caffe_copy(dim, bottom_data + data_offset_bottom,
          top_data + data_offset_top);
      }
    }
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!axis_){
    NOT_IMPLEMENTED;
    return;
  }
  for (int i = 0; i < top.size(); ++i) {
    // bottom[last] is the selector and never needs backpropagation
    // so we can iterate over top vector because top.size() == bottom.size() -1
    int new_tops_num = top[i]->shape(axis_);  
    CHECK_EQ(indices_to_forward_.size(), new_tops_num);
    int c = bottom[i]->shape(axis_);
    const int dim = top[i]->count() / new_tops_num;
    const int hw = bottom[i]->shape(2) * bottom[i]->shape(3);
    const int chw = c * hw;
    const int cphw = new_tops_num * hw;
    int next_to_backward_offset = 0;
    int batch_offset;
    int data_offset_bottom;
    int data_offset_top;
    int zeroout;
    for (int n = 0; n < c; ++n) {
      data_offset_bottom = n * dim;
      zeroout = 0;
      if (next_to_backward_offset >= new_tops_num) {
        // we already visited all items that were been forwarded, so
        // just set to zero remaining ones
        zeroout = 1;
      } else {
        batch_offset = indices_to_forward_[next_to_backward_offset];
        if (n != batch_offset) {  // this data was not been forwarded
          zeroout = 1;        
        } else {  // this data was been forwarded
          if (axis_) {
                filter_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
                dim, top[i]->mutable_gpu_diff(), bottom[i]->mutable_gpu_diff(), next_to_backward_offset, n, hw, cphw, chw);
          }
          else {
                data_offset_top = next_to_backward_offset * dim;
                caffe_copy(dim, top[i]->mutable_gpu_diff() + data_offset_top,
                    bottom[i]->mutable_gpu_diff() + data_offset_bottom);
          }            
          ++next_to_backward_offset;  // point to next forwarded item index
        }
      }
      if (zeroout){
          if (axis_){
            filter_zero_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
            dim, bottom[i]->mutable_gpu_diff(), n, hw, chw);
          } else{          
            caffe_gpu_set(dim, Dtype(0),
                bottom[i]->mutable_gpu_diff() + data_offset_bottom);
          }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FilterLayer);

}  // namespace caffe
