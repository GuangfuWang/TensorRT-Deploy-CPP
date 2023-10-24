#pragma once

#include <string>
#include <vector>
#include <NvInfer.h>
#include "preprocessor.h"
#include "postprocessor.h"
#include "trt_deployresult.h"
#include "util.h"

namespace fight
{
/**
 * @brief this is a helper class for logging.
 * @details subclassing the ILogger is necessary.
 * @note this is wired because it is needed by TensorRT.
 */
class Logger: public nvinfer1::ILogger
{
public:
	/**
	 * @brief this is virtually implemented function.
	 * @param severity logging level.
	 * @param msg logging messages.
	 */
	void log(Severity severity, const char *msg) noexcept override;
};

/**
 * @brief This is main deploy class to invoke all inference functionalities.
 * @details To use the deploy class, you need first implement all preprocessor class and post processor class.
 * @note this class can be derived.
 */
class TrtDeploy
{
public:
	/**
	 * @brief constructor for deploy class.
	 * @details this construction will not init all materials.
	 */
	explicit TrtDeploy(SharedRef<Config>& config);

	/**
	 * @brief virtual de-constructor to avoid memory leaking.
	 * @details cuda memory need to be freed and others not.
	 */
	virtual ~TrtDeploy();

public:
	enum class ModelLoadStatus
	{
		LOADED_SUCCESS = 0,///< load the model successfully.
		LOADED_FAILED = 1,///< load the model failed, possibly not found or wrong model file.
		NON_LOADED = 2 ///< have not loaded model.
	};
	enum class CudaMemAllocStatus
	{
		ALLOC_SUCCESS = 0,///< allocation of memory on GPU side successfully.
		ALLOC_FAILED = 1,///< allocation of memory on GPU side failed, possibly out of memory.
		NON_ALLOC = 2 ///< have not allocated any memory.
	};

public:
	/**
	 * @brief This method is used as infer function, its input is images data.
	 * @param img inout images, for videos, it should be parsed.
	 * @param result inference results.
	 */
	virtual void Infer(const std::vector<cv::Mat> &img, SharedRef<TrtResults> &result);

	/**
	 * @brief inference for fake data.
	 * @details the main purpose of this function is to test the whole pipeline's capability.
	 * @param res inference results.
	 */
	virtual void Warmup(SharedRef<TrtResults> &res);

	/**
	 * @brief This is the post processing function, you can implement the real worker as Postprocessor object.
	 * @details
	 * @note it is the Post processing 's responsibility to unscale if images are scaled or pad.
	 * @param res inference results.
	 * @param img input images.
	 * @param out_img output images.
	 */
	void Postprocessing(const SharedRef<TrtResults> &res, const std::vector<cv::Mat> &img,
						std::vector<cv::Mat> &out_img, int& alarm);

protected:
	/**
	 * @brief internal infer function with gpu input.
	 * @param data input data.
	 * @param res output results.
	 */
	void InferResults(SharedRef<ImageBlob> &data, SharedRef<TrtResults> &res);

	/**
	 * @brief initialization of all necessary staff.
	 * @details including allocating memory on gpu / init objects / set input and output.
	 * @note the model file is the full path and it should be forward slashed.
	 * @warning the path should not contain any chinese characters.
	 * @param model_file algorithm model file.
	 */
	virtual void Init(const std::string &model_file);

	/**
	 * @brief get model load status.
	 * @return model loading status.
	 */
	ModelLoadStatus LoadStatus();

	/**
	 * @brief get cuda memory allocation status.
	 * @return cuda allocation status.
	 */
	CudaMemAllocStatus MemAllocStatus();

protected:
	bool INIT_FLAG = false; ///< to indicate the system has initialized.
	SharedRef<Preprocessor> m_preprocessor = nullptr; ///< preprocessor object.
	SharedRef<Postprocessor> m_postprocessor = nullptr; ///< post processor object.
	nvinfer1::ICudaEngine* m_engine = nullptr; ///< cuda engine object.
	nvinfer1::IRuntime* m_runtime = nullptr; ///< cuda runtime.
	nvinfer1::IExecutionContext* m_execution_context = nullptr; ///< cuda context.
	cudaStream_t m_stream = nullptr; ///< for parallel purpose.
	SharedRef<Logger> m_logger = nullptr; ///< logger util.
	std::vector<void *> m_device_ptr; ///< pointer to states on GPU side.
	std::vector<cv::cuda::GpuMat> m_cv_data;///< directly map from opencv GpuMat to TensorRT.

	std::vector<std::vector<float>> m_output_state; ///< output data.

	ModelLoadStatus m_model_load_status; ///< model loading status.
	CudaMemAllocStatus m_cuda_alloc_status; ///< allocation of memory for cuda.

	float m_curr_fps; ///< Frame per Second.

	SharedRef<Config> m_config = nullptr;
};

}