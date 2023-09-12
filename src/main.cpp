#include "config.h"
#include "trt_deploy.h"
#include "trt_deployresult.h"

using namespace gf;

/**
 * @example
 * @param argc number of input params, at least 1.
 * @param argv params lists
 * @return
 */

int main(int argc, char **argv)
{
	Config::LoadConfigFile(argc, argv, "../config/infer_cfg.yaml");
	SharedRef<TrtDeploy> mDeploy = createSharedRef<TrtDeploy>();
	SharedRef<TrtResults> mResult = createSharedRef<TrtResults>();
	mDeploy->Warmup(mResult);
	//prepare the input data.
	auto in_path = std::filesystem::path(Config::VIDEO_FILE);
	cv::VideoCapture cap(in_path);
	cv::VideoWriter vw;
	std::filesystem::path output_path = in_path.parent_path() / (in_path.stem().string() + ".result.mp4");
	vw.open(output_path,
			cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
			cap.get(cv::CAP_PROP_FPS),
			cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
	size_t total = cap.get(cv::CAP_PROP_FRAME_COUNT), current = 0;
	std::vector<cv::Mat> imgs;
	std::vector<cv::Mat> out_data;
	cv::Mat img;
	while (cap.read(img)) {
		imgs.emplace_back(img);
		++current;
		if (imgs.size() >= Config::TRIGGER_LEN) {
			mDeploy->Infer(imgs, mResult);
			mDeploy->Postprocessing(mResult, imgs, out_data);
			for (auto &f : out_data) {
				vw.write(f);
			}
			imgs.clear();
			out_data.clear();
		}
//		std::cout << "\r" << current << "/" << total << std::endl;
//		std::cout.flush();
	}
	return 0;
}