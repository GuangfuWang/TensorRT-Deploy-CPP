#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime_api.h>
#include <opencv4/opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

namespace gf
{


///@note the frame data should be in [N,C,H,W] order.
///todo: potentially improved by cudaStreams in preprocessing pipeline.
inline void ResizeOnGpu(cv::cuda::GpuMat *raw, cv::cuda::GpuMat *resized,
						const float &scale_h = 1.0, const float scale_w = 1.0, int FLAG = cv::INTER_LINEAR)
{
	cv::cuda::resize(*raw, *resized, cv::Size(),
					 scale_w, scale_h, FLAG);
}

inline void ResizeOnGpu(cv::cuda::GpuMat *raw, cv::cuda::GpuMat *resized, cv::cuda::Stream &stream,
						const unsigned int &num = 1, const float &scale_h = 1.0, const float scale_w = 1.0,
						int FLAG = cv::INTER_LINEAR)
{
	for (int i = 0; i < num; ++i) {
		cv::cuda::resize(*(raw + i), *(resized + i), cv::Size(),
						 scale_w, scale_h, FLAG, stream);
	}
	stream.waitForCompletion();
}

inline void PadOnGpu(cv::cuda::GpuMat *raw, cv::cuda::GpuMat *padded, cv::cuda::Stream &stream,
					 const int &top, const int bottom, const int &left, const int &right,
					 const int &borderType = cv::BORDER_CONSTANT, const cv::Scalar &scalar = cv::Scalar(0),
					 const unsigned int &num = 1)
{
	for (int i = 0; i < num; ++i) {
		cv::cuda::copyMakeBorder(*(raw + i), *(padded + i), top, bottom, left, right,
								 borderType, scalar, stream);
	}
	stream.waitForCompletion();
}

///todo: potentially improved by block-wise and more stream involved.
inline void CvtColorOnGpu(cv::cuda::GpuMat *raw, cv::cuda::GpuMat *cvt, cv::cuda::Stream &stream,
						  const unsigned int &num, const cv::ColorConversionCodes &code)
{
	for (int i = 0; i < num; ++i) {
		cv::cuda::cvtColor(*(raw + i), *(cvt + i), code, 0, stream);
	}
	stream.waitForCompletion();
}

inline void ExtractChannelOnGpu(const cv::cuda::GpuMat *raw,
								std::vector<cv::cuda::GpuMat *> &cvt, cv::cuda::Stream &stream,
								const unsigned int &num)
{
	cvt.resize(num);
	for (int i = 0; i < num; ++i) {
		cv::cuda::split(*(raw + i), cvt[i], stream);
	}
	stream.waitForCompletion();
}

inline void ExtractChannelOnGpu(const cv::cuda::GpuMat *raw, std::vector<std::vector<cv::cuda::GpuMat>> &cvt,
								cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
								const unsigned int &num = 1)
{
	cvt.resize(num);
	for (int i = 0; i < num; ++i) {
		cvt[i].resize(raw->channels());//3 channels;
		cv::cuda::split(*(raw + i), cvt[i], stream);
	}
	stream.waitForCompletion();
}

inline void ConvertToOnGpu(cv::cuda::GpuMat *raw, int FLAG)
{
	const double e = 0.00392157; //1.0/255.0
	(*raw).convertTo(*raw, FLAG, e);
}

inline void NormalizeImageOnGpu(cv::cuda::GpuMat *raw, cv::cuda::Stream &stream, const unsigned int &num,
								const cv::cuda::GpuMat &scale_mat, const cv::cuda::GpuMat &add_mat
)
{
	for (int i = 0; i < num; ++i) {
		cv::cuda::multiply(*(raw + i), scale_mat, *(raw + i), 1.0, -1, stream);
	}
	stream.waitForCompletion();
	for (int i = 0; i < num; ++i) {
		cv::cuda::add(*(raw + i), add_mat, *(raw + i), cv::noArray(), -1, stream);
	}
	stream.waitForCompletion();
}
}