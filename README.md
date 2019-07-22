# ExtractFeatures
This code is feature extraction code based on Caffe, C++.
Input: image (OpenCV cv::Mat)
Output: features (std::vector<float>)

# Usage
Download BVLC caffe (https://github.com/BVLC/caffe)

copy the extract_features_custom.cpp to caffe-master/tools/ path

Edit Makefile.config (ref. http://caffe.berkeleyvision.org/installation.html)

make all
make test
make runtest





