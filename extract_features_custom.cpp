#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include <caffe/caffe.hpp>
#include <caffe/data_transformer.hpp>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;

void cvMatToDatum(const cv::Mat& _img, caffe::Datum* _datum)
{
	CHECK(_img.depth() == CV_8U) << "Image data type must be unsigned byte";
	_datum->set_channels(_img.channels());
	_datum->set_height(_img.rows);
	_datum->set_width(_img.cols);
	_datum->clear_data();
	_datum->clear_float_data();
	_datum->set_encoded(false);
	int datum_channels = _datum->channels();
	int datum_height = _datum->height();
	int datum_width = _datum->width();
	int datum_size = datum_channels * datum_height * datum_width;
	std::string buffer(datum_size, ' ');
	for (int h = 0; h < datum_height; ++h) {
		const uchar* ptr = _img.ptr<uchar>(h);
		int img_index = 0;
		for (int w = 0; w < datum_width; ++w) {
			for (int c = 0; c < datum_channels; ++c) {
				int datum_index = (c * datum_height + h) * datum_width + w;
				buffer[datum_index] = static_cast<char>(ptr[img_index++]);
			}
		}
	}
	_datum->set_data(buffer);
}

void cvMatToDatum(const cv::Mat &_img, caffe::Datum *_datum, int _depth)
{
    if (_depth == CV_8U)
    {
        cvMatToDatum(_img, _datum);
        return;
    }

    CHECK(_img.depth() == CV_32F) << "data type must be 32bit float";

    _datum->set_channels(_img.channels());
    _datum->set_height(_img.rows);
    _datum->set_width(_img.cols);
    _datum->clear_data();
    _datum->clear_float_data();
    _datum->set_encoded(false);
    int datum_channels = _datum->channels();
    int datum_height = _datum->height();
    int datum_width = _datum->width();
    int datum_size = datum_channels * datum_height * datum_width;

    for (int h = 0; h < datum_height; ++h)
    {
        const float *ptr = _img.ptr<float>(h);
        int img_index = 0;
        for (int w = 0; w < datum_width; ++w)
        {
            for (int c = 0; c < datum_channels; ++c)
            {
                _datum->add_float_data(ptr[img_index++]);
            }
        }
    }
    return;
}

template <typename Dtype>
int feature_extraction_pipeline(int argc, char **argv);

int main(int argc, char **argv)
{
    return feature_extraction_pipeline<float>(argc, argv);
    //  return feature_extraction_pipeline<double>(argc, argv);
}

template <typename Dtype>
int feature_extraction_pipeline(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);
    const int num_required_args = 8;
    if (argc < num_required_args)
    {
        LOG(ERROR) << "This program takes in a trained network and an input data layer, and then"
                      " extract features of the input data produced by the net.\n"
                      "Usage: extract_features_custom  pretrained_net_param"
                      "  feature_extraction_proto_file  extract_feature_blob_name"
                      "  save_dir  image_list_file mean_type[value/file] mean_data[mean_value[R,G,B]/mean_file]"
                      "  [CPU/GPU] [DEVICE_ID=0]\n"
                      "Note: you can extract multiple features in one pass by specifying"
                      " multiple feature blob names and dataset names separated by ','."
                      " The names cannot contain white space characters and the number of blobs"
                      " and datasets must be equal.";
        return 1;
    }
    int arg_pos = num_required_args;

    arg_pos = num_required_args;
    if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0)
    {
        LOG(ERROR) << "Using GPU";
        int device_id = 0;
        if (argc > arg_pos + 1)
        {
            device_id = atoi(argv[arg_pos + 1]);
            CHECK_GE(device_id, 0);
        }
        LOG(ERROR) << "Using Device_id=" << device_id;
        Caffe::SetDevice(device_id);
        Caffe::set_mode(Caffe::GPU);
    }
    else
    {
        LOG(ERROR) << "Using CPU";
        Caffe::set_mode(Caffe::CPU);
    }

    arg_pos = 0; // the name of the executable
    std::string pretrained_binary_proto(argv[++arg_pos]);

    // Expected prototxt contains at least one data layer such as
    //  the layer data_layer_name and one feature blob such as the
    //  fc7 top blob to extract features.
    /*
    input: "data"
    input_dim: 1
    input_dim: 3
    input_dim: 224
    input_dim: 224
    layer {
        name: "conv1_1"
        type: "Convolution"
        bottom: "data"
        top: "conv1_1"
        convolution_param {
            num_output: 64
            pad: 1
            kernel_size: 3
        }
    }
    layer {
        name: "fc7"
        type: "InnerProduct"
        bottom: "fc6"
        top: "fc7"
        inner_product_param {
            num_output: 4096
        }
    }
   */
    std::string feature_extraction_proto(argv[++arg_pos]);
    std::string blob_name(argv[++arg_pos]);
    std::string save_feature_dataset_dir(argv[++arg_pos]);
    if (save_feature_dataset_dir[save_feature_dataset_dir.size() - 1] != '/') {
        save_feature_dataset_dir += "/";
    }
    std::string image_list_file(argv[++arg_pos]);

    bool use_mean_file = false;
    
	caffe::TransformationParameter trans_para;
    if (strcmp(argv[++arg_pos], "value") == 0)
    {
        LOG(ERROR) << "Using mean valuse";
        use_mean_file = false;
        std::vector<std::string> values;
        std::string mean_values(argv[++arg_pos]);
        boost::split(values, mean_values, boost::is_any_of(","));
        CHECK_EQ(values.size(), 3) <<
        "the number of mean values must be three";
        trans_para.add_mean_value(stof(values[0]));
        trans_para.add_mean_value(stof(values[1]));
        trans_para.add_mean_value(stof(values[2]));
    }
    else
    {
        LOG(ERROR) << "Using mean file";
        use_mean_file = true;
        trans_para.set_mean_file(argv[++arg_pos]);
    }
    boost::shared_ptr<caffe::DataTransformer<float>> data_transformer(new caffe::DataTransformer<float>(trans_para, caffe::TEST));

    boost::shared_ptr<Net<Dtype>> feature_extraction_net(
        new Net<Dtype>(feature_extraction_proto, caffe::TEST));
    feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);
    
    CHECK_EQ(feature_extraction_net->num_inputs(), 1) << "Network should have exactly one input.";
    const std::vector<caffe::Blob<float>*> net_input = feature_extraction_net->input_blobs();

    CHECK(feature_extraction_net->has_blob(blob_name))
        << "Unknown feature blob name " << blob_name
        << " in the network " << feature_extraction_proto;

    LOG(ERROR) << "Load image file list";
    
    std::ifstream f_in(image_list_file.data());
    std::vector<std::string> images;
    if (f_in.is_open()) 
    {
        std::string line;
        while(getline(f_in, line)) {
            images.push_back(line);
        }
        f_in.close();
    }

    LOG(ERROR) << "Extracting Features & Save features";
    std::string save_file = save_feature_dataset_dir + blob_name + "_extracted_features.txt";
    std::ofstream f_out(save_file.data());

    caffe::Blob<float>* input_layer = feature_extraction_net->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();

    for (int i = 0; i < images.size(); ++i) 
    {
        std::vector<std::string> pasing;
        boost::split(pasing, images[i], boost::is_any_of(" "));

        // transform the image to datum which is applied mean values
        cv::Mat image = cv::imread(pasing[0]);
        cv::resize(image, image, cv::Size(width, height));
        caffe::Datum datum;
	    cvMatToDatum(image, &datum, image.depth());
        data_transformer->Transform(datum, net_input[0]);

        // extract the features from image
        feature_extraction_net->Forward();

        boost::shared_ptr<caffe::Blob<float>> output_layer = feature_extraction_net->blob_by_name(blob_name);
	    const float* data_begin = output_layer->cpu_data() + output_layer->offset(0);
	    const float* data_end = data_begin + output_layer->channels();

        std::vector<float> features(data_begin, data_end);

        // write the features to text file. >>path label feature[0] feature[1] ..... feature[n]\n
        if (f_out.is_open()) {
            f_out << images[i] << " ";
            for (int j = 0; j < features.size(); ++j) {
                if (j == 0) {
                    f_out << features[j];
                }
                else {
                    f_out << " " << features[j];
                }  
            }
            f_out << std::endl;
        }

        if (i % 1000 == 0) {
            LOG(ERROR) << "Current status: " << i;
        }
    }
    f_out.close();

    LOG(ERROR) << "Successfully extracted the features!";
    return 0;
}
