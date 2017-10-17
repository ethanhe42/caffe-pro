#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    if (this->layer_param_.convolution_param().has_filterwise_ratio()) {
      const float filterwiser_data = this->layer_param_.convolution_param().filterwise_ratio();
      const int count = top[i]->count();
      const int channels=this->channels_;
      const int output_channels=top[i]->shape(1);
      const int output_height=top[i]->shape(2); 
      const int output_width=top[i]->shape(3);
      const int input_height=bottom[i]->shape(2);
      const int input_width=bottom[i]->shape(3);
      const int kernel_height=this->blobs_[0]->shape(2); 
      const int kernel_width=this->blobs_[0]->shape(3);
      const int stride_y=this->stride_.cpu_data()[0]; 
      const int stride_x=this->stride_.cpu_data()[1]; 
      const int pad_y=this->pad_.cpu_data()[0]; 
      const int pad_x=this->pad_.cpu_data()[1];
      int ow, indexer, oh, n, iw, ih, wcoff,xcoff;
      int hw= output_width*output_height;
      int ihw= input_height * input_width;
      int khw= kernel_width * kernel_height;
      int chw = hw * output_channels;
      int exp_ch = filterwiser_data * channels;
      for (int index=0; index < count; index++) {
        ow = index % output_width;
        indexer = index/output_width;
        oh = indexer % output_height;
        indexer /= output_height;
        ///int c = indexer % channels;
        n = indexer / output_channels;
        //n = index / chw;
        iw = ow * stride_x - pad_x;
        ih = oh * stride_y - pad_y;

        Dtype v = 0;
        int woff = 0;
        int xoff = 0;
        for (int kh = 0; kh < kernel_height; kh++) {
          if (ih + kh >= 0 && ih + kh < input_height) {
            for (int kw = 0; kw < kernel_width; kw++) {
              if (iw + kw >= 0 && iw + kw < input_width) {
                for (int ic = 0; ic < exp_ch; ic++){
                  xcoff = ic * input_height * input_width + ((n * channels) * input_height + ih) * input_width + iw;
                  wcoff = ic * kernel_width * kernel_height;
                  //xcoff = ic * ihw;
                  //wcoff = ic * khw;
                  //v += (bottom_data+xoff+xcoff)[kw] * (weight+woff+wcoff)[kw];
                  //v += bottom_data[xoff+xcoff+kw] * weight[woff+wcoff+kw];
                  *(top_data+index) += bottom_data[xoff+xcoff+kw] * weight[woff+wcoff+kw];

                }
              }
            }
          }
          xoff += input_width;
          woff += kernel_width;
        }
        //*(top_data+index) = v;
      }
    }
    else{
      for (int n = 0; n < this->num_; ++n) {
        this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->cpu_data();
          this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
