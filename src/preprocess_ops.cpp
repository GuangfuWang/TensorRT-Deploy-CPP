
#include "preprocess_ops.h"
#include "preprocess_util.hpp"
#include "preprocessor.h"

namespace fight
{

void NormalizeImage::Run(std::vector<cv::cuda::GpuMat> &data)
{
	if(!init){
		init = true;
		const auto ori_w = m_config->TARGET_SIZE[1];
		const auto ori_h = m_config->TARGET_SIZE[0];
		const float NORM_CONST = 0.00392157f;//1.0/255.0
		//init the matrix.
		const float coe_1 = NORM_CONST / m_config->N_STD[0];
		const float coe_2 = NORM_CONST / m_config->N_STD[1];
		const float coe_3 = NORM_CONST / m_config->N_STD[2];
		const float off_1 = -m_config->N_MEAN[0] / m_config->N_STD[0];
		const float off_2 = -m_config->N_MEAN[1] / m_config->N_STD[1];
		const float off_3 = -m_config->N_MEAN[2] / m_config->N_STD[2];
		m_mul = cv::cuda::GpuMat(ori_h, ori_w, CV_32FC3,
									  cv::Scalar(coe_1, coe_2, coe_3));
		m_sub = cv::cuda::GpuMat(ori_h, ori_w, CV_32FC3,
										   cv::Scalar(off_1, off_2, off_3));
	}
	NormalizeImageOnGpu(data.data(), *m_stream, data.size(),
						m_mul, m_sub);
}

void Permute::Run(std::vector<cv::cuda::GpuMat> &data)
{
	///@note this method essentially extract each channel of a image and put it to an array.
	///@note such that: array <=> [C,H,W]
	///@note this ops has been done in CvtFromGpuMat, thus this will do nothing for gpu version.
}

void Resize::Run(std::vector<cv::cuda::GpuMat> &data)
{
	auto scale = GenerateScale(data[0]);
	if(std::abs(scale.first-1.0f)<1e-8||std::abs(scale.second-1.0f)<1e-8)return;
	ResizeOnGpu(data.data(), data.data(), *m_stream, data.size(),
				scale.first, scale.second, (int)m_config->INTERP);

}

std::pair<float, float> Resize::GenerateScale(const cv::cuda::GpuMat &im)
{
	std::pair<float, float> resize_scale;
	int origin_w = im.cols;
	int origin_h = im.rows;

	if (m_config->KEEP_RATIO) {
		int im_size_max = std::max(origin_w, origin_h);
		int im_size_min = std::min(origin_w, origin_h);
		int target_size_max =
			*std::max_element(m_config->TARGET_SIZE.begin(), m_config->TARGET_SIZE.end());
		int target_size_min =
			*std::min_element(m_config->TARGET_SIZE.begin(), m_config->TARGET_SIZE.end());
		float scale_min =
			static_cast<float>(target_size_min) / static_cast<float>(im_size_min);
		float scale_max =
			static_cast<float>(target_size_max) / static_cast<float>(im_size_max);
		float scale_ratio = std::min(scale_min, scale_max);
		resize_scale = {scale_ratio, scale_ratio};
	}
	else {
		//always [H,W] order.
		resize_scale.first =
			static_cast<float>(m_config->TARGET_SIZE[0]) / static_cast<float>(origin_h);

		resize_scale.second =
			static_cast<float>(m_config->TARGET_SIZE[1]) / static_cast<float>(origin_w);

	}

	return resize_scale;
}

void LetterBoxResize::Run(std::vector<cv::cuda::GpuMat> &data)
{
	float resize_scale = GenerateScale(data[0]);
	auto new_shape_w = (int)std::round((float)data[0].cols * resize_scale);
	auto new_shape_h = (int)std::round((float)data[0].rows * resize_scale);

	auto pad_w = (float)(m_config->TARGET_SIZE[1] - new_shape_w) / 2.0f;
	auto pad_h = (float)(m_config->TARGET_SIZE[0] - new_shape_h) / 2.0f;

	int top = (int)std::round(pad_h - 0.1);
	int bottom = (int)std::round(pad_h + 0.1);
	int left = (int)std::round(pad_w - 0.1);
	int right = (int)std::round(pad_w + 0.1);
	if(new_shape_w!=data[0].cols||new_shape_h!=data[0].rows){
		ResizeOnGpu(data.data(), data.data(), *m_stream, data.size(),
					(float)new_shape_h / (float)data[0].rows, (float)new_shape_w / (float)data[0].cols,
					cv::INTER_AREA);
	}

	PadOnGpu(data.data(), data.data(), *m_stream,
			 top, bottom, left, right,
			 cv::BORDER_CONSTANT, cv::Scalar(127.5), data.size());
}

float LetterBoxResize::GenerateScale(const cv::cuda::GpuMat &im)
{
	int origin_w = im.cols;
	int origin_h = im.rows;

	int target_h = m_config->TARGET_SIZE[0];
	int target_w = m_config->TARGET_SIZE[1];

	float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
	float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
	float resize_scale = std::min(ratio_h, ratio_w);
	return resize_scale;
}

void PadStride::Run(std::vector<cv::cuda::GpuMat> &data)
{
	const int s = (int)m_config->STRIDE;
	if (s <= 0)return;

	int rh = data[0].rows;
	int rw = data[0].cols;
	int nh = (rh / s) * s + (rh % s != 0) * s;
	int nw = (rw / s) * s + (rw % s != 0) * s;
	PadOnGpu(data.data(), data.data(), *m_stream, 0, nh - rh, 0, nw - rw,
			 cv::BORDER_CONSTANT, cv::Scalar(0), data.size());
}

void TopDownEvalAffine::Run(std::vector<cv::cuda::GpuMat> &data)
{
	ResizeOnGpu(data.data(), data.data(), *m_stream, data.size(), (float)m_config->TRAIN_SIZE[0] / (float)data[0].rows,
				(float)m_config->TRAIN_SIZE[1] / (float)data[0].cols, (int)m_config->INTERP);
}
}
