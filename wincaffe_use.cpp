//2017-07-20 Edit by KMC
#include "wincaffe_use.h"

std::vector<float> WinCaffeUseClass::extract_features(cv::Mat& _img)
{
	clock_t before;
	double r;

	// Check the given blob name
	CHECK(m_pCaffeNet->has_blob(m_strBlobName))         // check existence of blob
		<< "Unknown feature blob name " << m_strBlobName
		<< " in the network " << m_pCaffeNet;

	//cv::Mat 영상을 Datum 데이터로 변환 
	caffe::Datum datum;
	cvMatToDatum(_img, &datum, _img.depth());

	//입력값 셋팅  
	boost::shared_ptr<caffe::DataTransformer<float>> data_transformer;
	caffe::TransformationParameter trans_para;
	//Mean Value 셋팅
	if (m_bMeanValue)
	{
		trans_para.add_mean_value(m_fMeanValue[0]);
		trans_para.add_mean_value(m_fMeanValue[1]);
		trans_para.add_mean_value(m_fMeanValue[2]);
	}
	else
	{
		trans_para.set_mean_file(m_strMeanFile);
	}

	/* Divice ID Set. */
	int device_id = 0;
	CHECK_GE(device_id, 0);
	caffe::Caffe::SetDevice(device_id);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	// 위에 설정한 셋팅대로 DataTransformer를 통해 입력 셋팅 
	data_transformer.reset(new caffe::DataTransformer<float>(trans_para, caffe::TEST));
	const std::vector<caffe::Blob<float>*> net_input = m_pCaffeNet->input_blobs();
	// 입력되는 값 영상 사이즈 Net에 맞게 변경하여 입력되야함
	data_transformer->Transform(datum, net_input[0]);

	before = clock();
	m_pCaffeNet->Forward();
	r = (double)(clock() - before) / CLOCKS_PER_SEC;
	std::cout << "Forward(s): " << r << std::endl;

	//결과 출력 
	boost::shared_ptr<caffe::Blob<float>> output_layer = m_pCaffeNet->blob_by_name(m_strBlobName);

	const float* data_begin = output_layer->cpu_data() + output_layer->offset(0);
	const float* data_end = data_begin + output_layer->channels();

	return std::vector<float>(data_begin, data_end);
}

void WinCaffeUseClass::resetCaffeNet()
{
	/* Load the network. */
	m_pCaffeNet.reset(new caffe::Net<float>(m_strCaffePrototxt, caffe::TEST));
	m_pCaffeNet->CopyTrainedLayersFrom(m_strCaffeModel);

	CHECK_EQ(m_pCaffeNet->num_inputs(), 1) << "Network should have exactly one input.";

	caffe::Blob<float>* input_layer = m_pCaffeNet->input_blobs()[0];
	m_iChannels = input_layer->channels();
	CHECK(m_iChannels == 3 || m_iChannels == 1)
		<< "Input layer should have 1 or 3 channels.";
	m_sizeInput = cv::Size(input_layer->width(), input_layer->height());

	return;
}

void WinCaffeUseClass::resetCaffeNet(const std::string _prototxt, const std::string _caffemodel)
{
	m_strCaffePrototxt = _prototxt;
	m_strCaffeModel = _caffemodel;

	/* Load the network. */
	m_pCaffeNet.reset(new caffe::Net<float>(m_strCaffePrototxt, caffe::TEST));
	m_pCaffeNet->CopyTrainedLayersFrom(m_strCaffeModel);

	CHECK_EQ(m_pCaffeNet->num_inputs(), 1) << "Network should have exactly one input.";

	caffe::Blob<float>* input_layer = m_pCaffeNet->input_blobs()[0];
	m_iChannels = input_layer->channels();
	CHECK(m_iChannels == 3 || m_iChannels == 1)
		<< "Input layer should have 1 or 3 channels.";
	m_sizeInput = cv::Size(input_layer->width(), input_layer->height());

	return;
}

void WinCaffeUseClass::cvMatToDatum(const cv::Mat& _img, caffe::Datum* _datum, int _depth)
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
		const float* ptr = _img.ptr<float>(h);
		int img_index = 0;
		for (int w = 0; w < datum_width; ++w) {
			for (int c = 0; c < datum_channels; ++c) {
				_datum->add_float_data(ptr[img_index++]);
			}
		}
	}
	return;
}

void WinCaffeUseClass::cvMatToDatum(const cv::Mat& _img, caffe::Datum* _datum)
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