/*
  Copyright (c) 2023, Guangfu WANG
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
  * Neither the name of the <organization> nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY <copyright holder> ''AS IS'' AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL <copyright holder> BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include "util.h"
#include "macro.h"
#include "config.h"
#include "preprocess_ops.h"


namespace gf
{

extern const float NORM_CONST;

struct ImageBlob
{
	// image width and height, should be int
	//todo: this should be the shape after the scale but before the padding.
	std::vector<int>     im_shape_;
	// Buffer for image data after preprocessing
	std::vector<float>   im_data_;
	// in net data shape(after pad)
	std::vector<int>     in_net_shape_;
	// Evaluation image width and height
	// std::vector<float>  eval_im_size_f_;
	// Scale factor for image size to origin image size
	std::vector<float>   scale_factor_;
	// in net image after preprocessing
	std::vector<cv::Mat> in_net_im_;
};

class PreprocessorFactory
{
public:
	static void Init();

	PreprocessorFactory();

	virtual ~PreprocessorFactory();

	void Run(const std::vector<cv::Mat> &input, SharedRef<ImageBlob> &output);

	void CvtForGpuMat(const std::vector<cv::Mat> &input, cv::cuda::GpuMat *frames, int &num);

	void CvtFromGpuMat(const cv::cuda::GpuMat *imgs, SharedRef<ImageBlob> &blob,
					   const std::array<float, 2> &scale, const int &num);

public:
	//these two matrix is used for normalizing the frame image channel wise.
	static thread_local cv::cuda::GpuMat MUL_MATRIX;
	static thread_local cv::cuda::GpuMat SUBTRACT_MATRIX;
	///@note this CONFIG must be set before actually inferring.
	static thread_local bool             INIT_FLAG;
	static thread_local float            SCALE_W;
	static thread_local float            SCALE_H;
private:
	SharedRef<Factory<PreprocessOp>> m_ops    = nullptr;
	SharedRef<cv::cuda::Stream>      m_stream = nullptr;
};

///@note this is GPU version of preprocessing pipeline.
class Preprocessor
{
public:
	void Run(const std::vector<cv::Mat> &input, SharedRef<ImageBlob> &output);

private:
	SharedRef<PreprocessorFactory> m_preprocess_factory = nullptr;
};
}
