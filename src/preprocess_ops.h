#pragma once

#include <yaml-cpp/yaml.h>
#include <opencv4/opencv2/opencv.hpp>
#include "util.h"

namespace gf {
    /**
     * @brief Abstraction of preprocessing operation class, copied.
     * @details this class should be used inside the PreprocessorFactory class.
     */
    class PreprocessOp {
    public:
		/**
		 * @brief de-constructor.
		 */
        virtual ~PreprocessOp() = default;
		/**
		 * @brief the interface function, invoking from PreprocessorFactory class.
		 * @param data data from gpu side.
		 * @param num number of images.
		 */
        virtual void Run(std::vector<cv::cuda::GpuMat> &data) = 0;

    protected:
        SharedRef<cv::cuda::Stream> m_stream = createSharedRef<cv::cuda::Stream>();///< for parallel purpose.
    };

	/**
	 * @brief normalization of given image data on GPU side.
	 * @details channel wise mean and standard deviation should be given.
	 */
    class NormalizeImage final : public PreprocessOp {
    public:
		/**
		 * @brief implementation function.
		 * @param data image data, from a vector->data().
		 * @param num number of images.
		 */
        void Run(std::vector<cv::cuda::GpuMat> &data) override;
    };
	/**
	 * @brief do nothing, yeah yeah i know it is silly, this class is kept only to make somebody happy, --!>
	 * --> and make them think my implementation is exactly same as original one in pipeline.
	 */
    class Permute final : public PreprocessOp {
    public:
		/**
		 * @brief do nothing function, see
		 * @param data input image.
		 * @param num number of images.
		 */
        void Run(std::vector<cv::cuda::GpuMat> &data) override;
    };
	/**
	 * @brief resizing the images from GPU side, the parameter needed is CONFIG class.
	 */
    class Resize final : public PreprocessOp {
    public:
		/**
		 * @brief resize image according to CONFIG.
		 * @param data raw images
		 * @param num number of raw images.
		 */
        void Run(std::vector<cv::cuda::GpuMat> &data) override;

    private:
        static std::pair<float, float> GenerateScale(const cv::cuda::GpuMat &im);///<Compute best resize scale for x-dimension, y-dimension
    };

	/**
	 * @brief not used.
	 */
    class LetterBoxResize final : public PreprocessOp {
    public:
		/**
		 * @brief used for detection box resizing, not applied currently.
		 * @param data images.
		 * @param num number of images.
		 */
        void Run(std::vector<cv::cuda::GpuMat> &data) override;

    private:
		/// utility function to obtain scale.
        static float GenerateScale(const cv::cuda::GpuMat &im);
    };

	/**
	 * @brief padding operation worker class.
	 * @details Models with FPN need input shape % stride == 0
	 */
    class PadStride final : public PreprocessOp {
    public:
		/**
		 * @brief padding for data.
		 * @param data images.
		 * @param num number of images.
		 */
        void Run(std::vector<cv::cuda::GpuMat> &data) override;
    };

	/**
	 * @brief actually a resize class.
	 * @details need CONFIG::TRAIN_SIZE.
	 */
    class TopDownEvalAffine final : public PreprocessOp {
    public:
		/**
		 * @brief implementation function.
		 * @param data images.
		 * @param num number of images.
		 */
        void Run(std::vector<cv::cuda::GpuMat> &data) override;
    };
}
