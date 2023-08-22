#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>
#include "trt_deployresult.h"
#include "util.h"

namespace gf
{
class PostprocessorOps
{
public:
	virtual ~PostprocessorOps() = default;

	enum class PostProcessFlag
	{
		DRAW_LETTER     = 0,
		DRAW_BOX        = 1,
		DRAW_BOX_LETTER = 2,
		MASK_OUT        = 3,
	};

public:
	virtual void Run(const SharedRef<TrtResults> &res,
					 const std::vector<cv::Mat> &img,
					 std::vector<cv::Mat> &out_img) = 0;
};

class FightPpTSMDeployPost final: public PostprocessorOps
{
public:
	void Run(const SharedRef<TrtResults> &res, const std::vector<cv::Mat> &img,
			 std::vector<cv::Mat> &out_img) override;
};

class Postprocessor
{
public:
	virtual ~Postprocessor() = default;
	virtual void Run(const SharedRef<TrtResults> &res,
					 const std::vector<cv::Mat> &img,
					 std::vector<cv::Mat> &out_img);

	virtual void Init();

private:
	SharedRef<Factory<PostprocessorOps>> m_ops = createSharedRef<Factory<PostprocessorOps>>();
	static thread_local bool             INIT_FLAG;
};
}
