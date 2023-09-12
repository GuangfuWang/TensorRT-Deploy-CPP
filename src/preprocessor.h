#pragma once

#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include "util.h"
#include "macro.h"
#include "config.h"
#include "preprocess_ops.h"

namespace gf
{

extern const float NORM_CONST;///< normalization constant, should be 1.0/255.0.

/**
 * @brief utility struct for image data in-out to GPU.
 */
struct ImageBlob
{
	std::vector<int> im_shape_;///< image width and height, should be int
	std::vector<int> in_net_shape_;///< in net data shape(after pad)
	// Evaluation image width and height
	// std::vector<float>  eval_im_size_f_;
	std::vector<float> scale_factor_;///< Scale factor for image size to origin image size
	std::vector<cv::Mat> in_net_im_;///< in net image after preprocessing
	std::vector<cv::cuda::GpuMat> m_gpu_data;///< real gpu data.
};

/**
 * @brief this is factory class for preprocessing
 * @details this class contains all worker class for preprocessing purpose.
 * @note this class should not used inside deploy class, only use it inside the Preprocessor class.
 * @example:
 * @code
 * 	SharedRef<Factory<PreprocessOp>> m_ops;
 * 	if (!m_ops) {
 *		m_ops = createSharedRef<Factory<PreprocessOp>>();
 *	}
 *	m_ops->registerType<NormalizeImage>("NormalizeImage");
 *	m_ops->registerType<Permute>("Permute");
 *	m_ops->registerType<Resize>("Resize");
 *	m_ops->registerType<PadStride>("PadStride");
 *	m_ops->registerType<TopDownEvalAffine>("TopDownEvalAffine");
 *
 *	//prepare the gpu data.
 *	for (const auto &i : Config::PIPELINE_TYPE) {
 *		(m_ops->create(i))->Run(gpu, num);
 *	}
 * @endcode
 */
class PreprocessorFactory
{
public:
	/**
	 * @brief initialization of static staff.
	 */
	static void Init();
	/**
	 * @brief constructor.
	 * @details init the m_ops/m_stream, and register all worker subclass.
	 */
	PreprocessorFactory();
	/**
	 * @brief destroy the factory map.
	 */
	virtual ~PreprocessorFactory();
	/**
	 * @brief actual working function.
	 * @param input raw image data.
	 * @param output preprocessing results.
	 */
	void Run(const std::vector<cv::Mat> &input, SharedRef<ImageBlob> &output);

private:
	/**
 	* @brief convert cpu mat to GPU mat pointer
 	* @param input input raw data.
 	* @param frames pointer to GpuMat.
 	* @param num number of GpuMats.
 	*/
	void CvtForGpuMat(const std::vector<cv::Mat> &input, std::vector<cv::cuda::GpuMat>& frames, int &num);

public:
	//these two matrix is used for normalizing the frame image channel wise.
	static thread_local cv::cuda::GpuMat MUL_MATRIX;///< used for normalization.
	static thread_local cv::cuda::GpuMat SUBTRACT_MATRIX;///< used for channel wise normalization.
	///@note this CONFIG must be set before actually inferring.
	static thread_local bool INIT_FLAG;///< indicate initialization status.
	static thread_local float SCALE_W;///< indicate scale of width.
	static thread_local float SCALE_H;///< indicate scale of height.
private:
	SharedRef<Factory<PreprocessOp>> m_ops = nullptr;///< worker smart pointer.
	SharedRef<cv::cuda::Stream> m_stream = nullptr;///< parallel support.
};

/**
 * @brief this is interface class for deploy class.
 * @details to conveniently use preprocessing functionality, only invoke Run() method is required.
 * @note this is GPU version of preprocessing pipeline.
 */
class Preprocessor
{
public:
	/**
	 * @brief invoking interface function for preprocessing by deploy class.
	 * @param input raw image data.
	 * @param output output preprocessed data.
	 */
	void Run(const std::vector<cv::Mat> &input, SharedRef<ImageBlob> &output);

private:
	SharedRef<PreprocessorFactory> m_preprocess_factory = nullptr;///< worker factory.
};
}
