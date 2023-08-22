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

#include <string>
#include <vector>
#include <NvInfer.h>
#include "preprocessor.h"
#include "postprocessor.h"
#include "trt_deployresult.h"
#include "util.h"

namespace gf
{

class Logger: public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char *msg) noexcept override;
};

class TrtDeploy
{
public:
	TrtDeploy();

	virtual ~TrtDeploy();

public:
	enum class ModelLoadStatus
	{
		LOADED_SUCCESS = 0,
		LOADED_FAILED  = 1,
		NON_LOADED     = 2
	};
	enum class CudaMemAllocStatus
	{
		ALLOC_SUCCESS = 0,
		ALLOC_FAILED  = 1,
		NON_ALLOC     = 2
	};

public:
	virtual void Infer(const std::vector<cv::Mat> &img, SharedRef<TrtResults> &result);

	virtual void Warmup(SharedRef<TrtResults> &res);

	///@note it is the Post processing 's responsibility to unscale if images are scaled or pad.
	void Postprocessing(const SharedRef<TrtResults> &res, const std::vector<cv::Mat> &img,
						std::vector<cv::Mat> &out_img);

protected:
	void InferResults(SharedRef<ImageBlob> &data, SharedRef<TrtResults> &res);

	virtual void Init(const std::string &model_file);

	ModelLoadStatus LoadStatus();

	CudaMemAllocStatus MemAllocStatus();

protected:
	static thread_local bool               INIT_FLAG;
	SharedRef<Preprocessor>                m_preprocessor      = nullptr;
	SharedRef<Postprocessor>               m_postprocessor     = nullptr;
	SharedRef<nvinfer1::ICudaEngine>       m_engine            = nullptr;
	SharedRef<nvinfer1::IRuntime>          m_runtime           = nullptr;
	SharedRef<nvinfer1::IExecutionContext> m_execution_context = nullptr;
	cudaStream_t                           m_stream            = nullptr;
	SharedRef<Logger>                      m_logger            = nullptr;
	std::vector<void *>                    m_device_ptr;

	std::vector<float>              m_input_state;
	std::vector<std::vector<float>> m_output_state;

	ModelLoadStatus    m_model_load_status;
	CudaMemAllocStatus m_cuda_alloc_status;

	float m_curr_fps;
};

}