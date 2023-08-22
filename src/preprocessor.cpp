#include "preprocess_util.hpp"
#include "preprocessor.h"

namespace gf
{
const float NORM_CONST = 0.00392157f;//1.0/255.0

#ifdef FIGHT_PREPROCESS_GPU

thread_local cv::cuda::GpuMat PreprocessorFactory::MUL_MATRIX = cv::cuda::GpuMat();

thread_local cv::cuda::GpuMat PreprocessorFactory::SUBTRACT_MATRIX = cv::cuda::GpuMat();

thread_local bool  PreprocessorFactory::INIT_FLAG = false;

thread_local float PreprocessorFactory::SCALE_W = 1.0f;

thread_local float PreprocessorFactory::SCALE_H = 1.0f;

void PreprocessorFactory::Init()
{
	assert(Config::N_STD[0] > 0.0f && Config::N_STD[1] > 0.0f && Config::N_STD[2] > 0.0f);

	auto       s     = Config::INPUT_SHAPE.size();
	const auto ori_w = Config::INPUT_SHAPE[s - 1];
	const auto ori_h = Config::INPUT_SHAPE[s - 2];
	SCALE_W = (float)Config::TARGET_SIZE[1] / (float)ori_w;
	SCALE_H = (float)Config::TARGET_SIZE[0] / (float)ori_h;
	//init the matrix.
	const float coe_1 = NORM_CONST / Config::N_STD[0];
	const float coe_2 = NORM_CONST / Config::N_STD[1];
	const float coe_3 = NORM_CONST / Config::N_STD[2];
	const float off_1 = -Config::N_MEAN[0] / Config::N_STD[0];
	const float off_2 = -Config::N_MEAN[1] / Config::N_STD[1];
	const float off_3 = -Config::N_MEAN[2] / Config::N_STD[2];
	MUL_MATRIX      = cv::cuda::GpuMat(ori_h, ori_w, CV_32FC3,
									   cv::Scalar(coe_1, coe_2, coe_3));
	SUBTRACT_MATRIX = cv::cuda::GpuMat(ori_h, ori_w, CV_32FC3,
									   cv::Scalar(off_1, off_2, off_3));
	INIT_FLAG       = true;
}

PreprocessorFactory::PreprocessorFactory()
{
	if (!m_ops) {
		m_ops = createSharedRef<Factory<PreprocessOp>>();
	}
	m_ops->registerType<NormalizeImage>("NormalizeImage");
	m_ops->registerType<Permute>("Permute");
	m_ops->registerType<Resize>("Resize");
	m_ops->registerType<PadStride>("PadStride");
	m_ops->registerType<TopDownEvalAffine>("TopDownEvalAffine");
	if (!m_stream) {
		m_stream = createSharedRef<cv::cuda::Stream>();
	}
}

void PreprocessorFactory::CvtFromGpuMat(const cv::cuda::GpuMat *imgs, SharedRef<ImageBlob> &out,
										const std::array<float, 2> &scale, const int &num)
{
	std::vector<std::vector<cv::cuda::GpuMat>> dev;
	std::vector<std::vector<cv::Mat>>          host(num);
	ExtractChannelOnGpu(imgs, dev, *m_stream, num);
	for (int i = 0; i < num; ++i) {
		host[i].resize(imgs->channels());
		for (int j = 0; j < imgs->channels(); ++j) {
			dev[i][j].download(host[i][j], *m_stream);
		}
	}
	m_stream->waitForCompletion();
	const int plane = imgs->rows * imgs->cols;
	const int vol   = plane * imgs->channels();
	out->im_data_.resize(num * vol);
	for (int i = 0; i < num; ++i) {
		for (int j = 0; j < imgs->channels(); ++j) {
			memcpy(&out->im_data_[j * plane + i * vol], host[i][j].data, plane * sizeof(float));
		}
	}

	//otherwise using cache value since it will not change during processing.
	if (out->im_shape_.empty()) {
		out->im_shape_ = Config::INPUT_SHAPE;
	}
	if (out->in_net_shape_.empty()) {
		out->in_net_shape_ = out->im_shape_;
		out->in_net_shape_[out->im_shape_.size() - 1] = Config::TARGET_SIZE[1];
		out->in_net_shape_[out->im_shape_.size() - 2] = Config::TARGET_SIZE[0];
	}
	if (out->scale_factor_.empty()) {
		out->scale_factor_.resize(2);
		out->scale_factor_[0] = scale[0];
		out->scale_factor_[1] = scale[1];
	}

	out->in_net_im_.resize(num);
	for (int i = 0; i < num; ++i) {
		(*(imgs + i)).download(out->in_net_im_[i], *m_stream);
	}
	m_stream->waitForCompletion();
}

void PreprocessorFactory::CvtForGpuMat(const std::vector<cv::Mat> &input,
									   cv::cuda::GpuMat *frames, int &num)
{
	if (Config::INPUT_SHAPE[Config::INPUT_SHAPE.size() - 1] != input[0].cols) {
		Config::INPUT_SHAPE[Config::INPUT_SHAPE.size() - 1] = input[0].cols;
		SCALE_W = (float)Config::TARGET_SIZE[1] / (float)input[0].cols;
		std::cout << "Input shape width in config file is not same as data width..." << std::endl;
	}
	if (Config::INPUT_SHAPE[Config::INPUT_SHAPE.size() - 2] != input[0].rows) {
		Config::INPUT_SHAPE[Config::INPUT_SHAPE.size() - 2] = input[0].rows;
		SCALE_H = (float)Config::TARGET_SIZE[0] / (float)input[0].rows;
		std::cout << "Input shape height in config file is not same as data height..." << std::endl;
	}
	num = (int)input.size();
	std::vector<cv::cuda::GpuMat> temp(num);
	for (int                      i = 0; i < num; ++i) {
		temp[i].upload(input[i], *m_stream);
	}
	m_stream->waitForCompletion();
	for (int i = 0; i < num; ++i) {
		temp[i].convertTo(temp[i], cv::COLOR_BGR2RGB, *m_stream);
	}
	m_stream->waitForCompletion();
	frames = temp.data();
}

void PreprocessorFactory::Run(const std::vector<cv::Mat> &input, SharedRef<ImageBlob> &output)
{
	if (!INIT_FLAG) {
		PreprocessorFactory::Init();
		INIT_FLAG = true;
	}
	cv::cuda::GpuMat *gpu;
	int              num = 0;
	CvtForGpuMat(input, gpu, num);
	for (const auto &i : Config::PIPELINE_TYPE) {
		(m_ops->create(i))->Run(gpu, num);
	}
	CvtFromGpuMat(gpu, output, {SCALE_H, SCALE_W}, num);
}

PreprocessorFactory::~PreprocessorFactory()
{
	if (m_ops) {
		m_ops->destroy();
	}
}

void Preprocessor::Run(const std::vector<cv::Mat> &input, SharedRef<ImageBlob> &output)
{
	if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
		std::cerr << "Your OpenCV does not support CUDA!" << std::endl;
		std::cerr << "Please install CUDA version OpenCV! "
					 "See: https://towardsdev.com/installing-opencv-4-with-cuda-in-ubuntu-20-04-fde6d6a0a367"
				  << std::endl;
		exit(EXIT_FAILURE);
	}
	if (!m_preprocess_factory) {
		m_preprocess_factory = createSharedRef<PreprocessorFactory>();
	}
	m_preprocess_factory->Run(input, output);
}


#endif

}