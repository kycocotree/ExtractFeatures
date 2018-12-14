//2017-07-20 Edit by KMC
//Caffe를 이용하여 피쳐 뽑을때 사용하는 클래스 

#pragma once

#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include <caffe/caffe.hpp>
#include <caffe/data_transformer.hpp>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include "common.h"

class WinCaffeUseClass
{
private:
	boost::shared_ptr<caffe::Net<float>> m_pCaffeNet;
	std::string m_strBlobName;
	cv::Size m_sizeInput;
	int m_iChannels;
	bool m_bMeanValue;
	float* m_fMeanValue;
	std::string m_strMeanFile;
	std::string m_strCaffePrototxt;
	std::string m_strCaffeModel;
public:
	WinCaffeUseClass()
	{
		m_strBlobName = "fc7";
		m_bMeanValue = true;
		m_fMeanValue = new float[3];
		m_fMeanValue[0] = 129.1863;
		m_fMeanValue[1] = 104.7624;
		m_fMeanValue[2] = 93.5940;

		/* Divice ID Set. */
		int device_id = 0;
		CHECK_GE(device_id, 0);	
		caffe::Caffe::SetDevice(device_id);
		caffe::Caffe::set_mode(caffe::Caffe::GPU);

		/* Load the network. */
		m_strCaffePrototxt = "data\\vgg_face_finetune\\VGG_FACE_Dongguk_deploy.prototxt";
		m_strCaffeModel = "data\\vgg_face_finetune\\VGG_FACE_CCTV_FR_FS20_DB1_iter_30000.caffemodel";

		m_pCaffeNet.reset(new caffe::Net<float>(m_strCaffePrototxt, caffe::TEST));
		m_pCaffeNet->CopyTrainedLayersFrom(m_strCaffeModel);

		CHECK_EQ(m_pCaffeNet->num_inputs(), 1) << "Network should have exactly one input.";

		caffe::Blob<float>* input_layer = m_pCaffeNet->input_blobs()[0];
		m_iChannels = input_layer->channels();
		CHECK(m_iChannels == 3 || m_iChannels == 1)
			<< "Input layer should have 1 or 3 channels.";
		m_sizeInput = cv::Size(input_layer->width(), input_layer->height());
	}
	WinCaffeUseClass(const std::string& model_file,
					 const std::string& trained_file,
					 const std::string& blob_name) 
	{
		m_strBlobName = blob_name;
		m_bMeanValue = true;
		m_fMeanValue = new float[3];
		m_fMeanValue[0] = 129.1863;
		m_fMeanValue[1] = 104.7624;
		m_fMeanValue[2] = 93.5940;

		/* Divice ID Set. */
		int device_id = 0;
		CHECK_GE(device_id, 0);
		caffe::Caffe::SetDevice(device_id);
		caffe::Caffe::set_mode(caffe::Caffe::GPU);

		/* Load the network. */
		m_pCaffeNet.reset(new caffe::Net<float>(trained_file, caffe::TEST));
		m_pCaffeNet->CopyTrainedLayersFrom(model_file);

		CHECK_EQ(m_pCaffeNet->num_inputs(), 1) << "Network should have exactly one input.";

		caffe::Blob<float>* input_layer = m_pCaffeNet->input_blobs()[0];
		m_iChannels = input_layer->channels();
		CHECK(m_iChannels == 3 || m_iChannels == 1)
			<< "Input layer should have 1 or 3 channels.";
		m_sizeInput = cv::Size(input_layer->width(), input_layer->height());
	}
	~WinCaffeUseClass()
	{
		SAFE_DELETE_ARRAY(m_fMeanValue);
	}
	void setBlobName(std::string _blob)	{ m_strBlobName = _blob; }
	std::vector<float> extract_features(cv::Mat& _img);
	void setMeanValue(const float _b, const float _g, const float _r) { m_bMeanValue = true; m_fMeanValue[0] = _b; m_fMeanValue[1] = _g; m_fMeanValue[2] = _r; }
	void setMeanFile(const std::string _path)	{ m_bMeanValue = false; m_strMeanFile = _path; }
	void setPrototxt(const std::string _path)		{ m_strCaffePrototxt = _path; }
	void setCaffeModel(const std::string _path)		{ m_strCaffeModel = _path; }
	void resetCaffeNet();
	void resetCaffeNet(const std::string _prototxt, const std::string _caffemodel);
private:
	void cvMatToDatum(const cv::Mat& _img, caffe::Datum* _datum, int _depth);
	void cvMatToDatum(const cv::Mat& _img, caffe::Datum* _datum);
};


