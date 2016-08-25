#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/matmul_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MatmulLayerTest : public GPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MatmulLayerTest()
      : blob_bottom_a(new Blob<Dtype>(2, 3, 4, 5)),blob_bottom_b(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>(2, 2, 1, 1)) {

    blob_bottom_vec_.push_back(blob_bottom_a);
    blob_bottom_vec_.push_back(blob_bottom_b);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MatmulLayerTest() { delete blob_bottom_a;delete blob_bottom_b; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_a;
  Blob<Dtype>* const blob_bottom_b;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MatmulLayerTest, TestDtypesAndDevices);

// TYPED_TEST(MatmulLayerTest, TestForward) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   MatmulLayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   // Test sum
//   for (int i = 0; i < this->blob_top_->num(); ++i) {
//     for (int k = 0; k < this->blob_top_->channels(); ++k) {
//         if (i == k) {
//             EXPECT_FLOAT_EQ()
//         }

//         // Test exact values
//         Dtype scale = 0;
//         for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
//             scale += exp(this->blob_bottom_->data_at(i, j, k, l));
//         }
//         for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
//             EXPECT_GE(this->blob_top_->data_at(i, j, k, l) + 1e-4,
//                 exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
//                 << "debug: " << i << " " << j;
//             EXPECT_LE(this->blob_top_->data_at(i, j, k, l) - 1e-4,
//                 exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
//                 << "debug: " << i << " " << j;
//         }
//     }
//   }
// }

TYPED_TEST(MatmulLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
    FillerParameter filler_param;
    filler_param.set_std(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a);
    filler.Fill(this->blob_bottom_b);

    LayerParameter layer_param;
    MatmulLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
    GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);    
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);    

//   LayerParameter layer_param;
//   MatmulLayer<Dtype> layer(layer_param);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_);
}

}  // namespace caffe
