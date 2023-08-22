#include <opencv2/imgproc.hpp>
#include "postprocessor.h"
#include "config.h"

namespace gf
{

thread_local bool Postprocessor::INIT_FLAG = false;

void FightPpTSMDeployPost::Run(const SharedRef<TrtResults> &res, const std::vector<cv::Mat> &img,
							   std::vector<cv::Mat> &out_img)
{
	//our simple program will only draw letters on top of images.
	auto flag = static_cast<PostProcessFlag>(Config::POST_MODE);
	assert(flag == PostProcessFlag::DRAW_LETTER);
	out_img.resize(img.size());
	std::vector<float> fight_res;
	res->Get(Config::OUTPUT_NAMES[0], fight_res);
	std::stringstream text;
	text << "Fight Detected with Confidence: " << 100 * fight_res[0] << "%";
	for (int i = 0; i < img.size(); ++i) {
		out_img[i] = img[i].clone();
		///@note the putText method does not have GPU version since it quite slow running on GPU for per pixel ops.
		cv::putText(out_img[i], text.str(),
					cv::Point(Config::TEXT_OFF_X, Config::TEXT_OFF_Y),
					cv::FONT_HERSHEY_PLAIN, Config::TEXT_FONT_SIZE,
					cv::Scalar(Config::TEXT_COLOR[0], Config::TEXT_COLOR[1], Config::TEXT_COLOR[2]),
					(int)Config::TEXT_LINE_WIDTH);
	}

}

void Postprocessor::Init()
{
	m_ops->registerType<FightPpTSMDeployPost>(Config::POSTPROCESS_NAME);
}

void Postprocessor::Run(const SharedRef<TrtResults> &res, const std::vector<cv::Mat> &img,
						std::vector<cv::Mat> &out_img)
{
	if (!INIT_FLAG) {
		Init();
		INIT_FLAG = true;
	}
	m_ops->create(Config::POSTPROCESS_NAME)->Run(res, img, out_img);
}
}