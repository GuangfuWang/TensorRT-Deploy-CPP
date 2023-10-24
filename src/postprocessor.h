#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>
#include "trt_deployresult.h"
#include "util.h"

namespace fight
{
/**
 * @brief This is a base class for real post processing.
 * @details This class should be implemented given the main function and the post processing purposes.
 * @note this class should NOT directly used by deploy class.
 * @note only the void Run(const SharedRef<TrtResults> &res, const std::vector<cv::Mat> &img, std::vector<cv::Mat> &out_img) need implement.
 */
class PostprocessorOps
{
public:
	explicit PostprocessorOps(SharedRef<Config>& config){
		m_config = config;
	}
	/**
	 * @brief virtual de-constructor for avoiding memory leaking.
	 */
	virtual ~PostprocessorOps() = default;

	enum class PostProcessFlag
	{
		DRAW_LETTER = 0, ///< indicate only draw text into image as inference results.
		DRAW_BOX = 1, ///< indicate only draw box into image as inference results.
		DRAW_BOX_LETTER = 2, ///< indicate draw both text and box as inference results.
		MASK_OUT = 3, ///< indicate mask out the box area.
		NON = 4,
	};

public:
	/**
	 * @brief This is main worker interface.
	 * @param res inference results.
	 * @param img raw images.
	 * @param out_img output images.
	 */
	virtual void Run(const SharedRef<TrtResults> &res,
					 const std::vector<cv::Mat> &img,
					 std::vector<cv::Mat> &out_img,int& alarm) = 0;

protected:
	SharedRef<Config> m_config = nullptr;
};

/**
 * @brief this is one of real implementation for base class PostprocessorOps.
 * @details this class is used for post processing the fight detection algorithm, specifically the video based one.
 * @note Do NOT use this class in deploy classes, only use this inside Postprocessor classes.
 */
class FightPpTSMDeployPost final: public PostprocessorOps
{
public:
	explicit FightPpTSMDeployPost(SharedRef<Config>& config): PostprocessorOps(config){};
	void Run(const SharedRef<TrtResults> &res, const std::vector<cv::Mat> &img,
			 std::vector<cv::Mat> &out_img,int & alarm) override;
private:
    std::vector<float> m_moving_average;///< moving average.
};

/**
 * @brief this is a post processing class used by deploy class.
 * @details This one should be used inside the deploy class, while the real worker should be a subclass of Ops.
 */
class Postprocessor final
{
public:
	explicit Postprocessor(SharedRef<Config>& config){m_config = config;}
	/**
 	* @brief de-constructor.
 	*/
	~Postprocessor();
	/**
	 * @brief invoking working function.
	 * @param res inference results.
	 * @param img raw images.
	 * @param out_img output images.
	 * @note the work is done using CPU computation, not GPU.
	 */
	void Run(const SharedRef<TrtResults> &res,
					 const std::vector<cv::Mat> &img,
					 std::vector<cv::Mat> &out_img,int& alarm);
private:
//	SharedRef<Factory<PostprocessorOps>> m_ops = nullptr;///< auto deconstructed, lazy purpose.
	PostprocessorOps* m_worker = nullptr;///< real worker.
	bool INIT_FLAG = false; ///< initialization flag.
	SharedRef<Config> m_config = nullptr;
};
}
