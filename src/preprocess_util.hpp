#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime_api.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>

namespace fight
{

/**
 *
 * @param raw input data
 * @param resized resized data.
 * @param scale_h scale factor for height.
 * @param scale_w scale factor for width.
 * @param FLAG resizing flag, @see /usr/local/include/opencv4/opencv2/imgproc.hpp
 * @note the frame data should be in [N,C,H,W] order.
 */
inline void ResizeOnGpu(cv::cuda::GpuMat *raw, cv::cuda::GpuMat *resized,
						const float &scale_h = 1.0, const float scale_w = 1.0, int FLAG = cv::INTER_LINEAR)
{
	cv::cuda::resize(*raw, *resized, cv::Size(),
					 scale_w, scale_h, FLAG);
}

/**
 *
 * @param raw raw GpuMat, should be from vector->data();
 * @param resized output GpuMat
 * @param stream for parallel purpose.
 * @param num number of input GpuMats.
 * @param scale_h scale factor for height.
 * @param scale_w scale factor for width.
 * @param FLAG resize flag for opencv.
 */
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

/**
 * @brief this is helper function for padding in GPU.
 * @param raw raw image data.
 * @param padded padded image data.
 * @param stream for parallel support.
 * @param top top row for padding.
 * @param bottom bottom row for padding.
 * @param left left column for padding.
 * @param right right column for padding.
 * @param borderType border padding type.
 * @param scalar padding value.
 * @param num number of input images.
 */
inline void PadOnGpu(cv::cuda::GpuMat *raw, cv::cuda::GpuMat *padded, cv::cuda::Stream &stream,
					 const int &top, const int &bottom, const int &left, const int &right,
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
/**
 * @brief this is a helper function for converting BGR to RGB.
 * @param raw input data.
 * @param cvt converted data.
 * @param stream parallel support.
 * @param num number of images.
 * @param code conversion code.
 */
inline void CvtColorOnGpu(cv::cuda::GpuMat *raw, cv::cuda::GpuMat *cvt, cv::cuda::Stream &stream,
						  const unsigned int &num, const cv::ColorConversionCodes &code)
{
	for (int i = 0; i < num; ++i) {
		cv::cuda::cvtColor(*(raw + i), *(cvt + i), code, 0, stream);
	}
	stream.waitForCompletion();
}

/**
 * @brief helper function for change images to pointers.
 * @param raw input data.
 * @param cvt converted data.
 * @param stream for parallel support.
 * @param num number of input images.
 */
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

/**
 * @brief extract channel wise data from GpuMat.
 * @param raw input iamges data.
 * @param cvt converted data.
 * @param stream parallel support.
 * @param num number of images.
 */
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

/**
 * @brief performing the channel wise normalization, the custom mean and deviation set by scale_mat and add_mat.
 * @param raw input data.
 * @param stream parallel support.
 * @param num number of images.
 * @param scale_mat scale matrix, normalization is equal to x --> (x-mu)/sigma, in which mu is mean and sigma is std.
 * @param add_mat add matrix, also is -mu/sigma matrix.
 */
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