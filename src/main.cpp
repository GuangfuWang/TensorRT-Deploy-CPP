#include "config.h"
#include "trt_deploy.h"
#include "trt_deployresult.h"
#include "model.h"

using namespace fight;

/**
 * @example
 * @param argc number of input params, at least 1.
 * @param argv params lists
 * @return
 */

int main(int argc, char **argv) {
    //prepare the input data.
    auto in_path = std::filesystem::path("/home/wgf/Downloads/datasets/fight/fight-mine/09-41-27.mp4");
    cv::VideoCapture cap(in_path);
    cv::VideoWriter vw;
    std::filesystem::path output_path = in_path.parent_path() / (in_path.stem().string() + ".result.mp4");
    vw.open(output_path,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            cap.get(cv::CAP_PROP_FPS),
            cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH),
					 cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

	bool init = false;
	cv::Mat img;
	cvModel* ptr = nullptr;
	cv::Mat curr;
    while (cap.isOpened()) {
		cap.read(img);
		if(img.empty())break;
		if(!init){
			ptr = Allocate_Algorithm(img,0,0);
			SetPara_Algorithm(ptr,0);
			UpdateParams_Algorithm(ptr);
			init = true;
		}
		curr = img.clone();
		Process_Algorithm(ptr,curr);
		std::cout<<"Current Res: "<<ptr->alarm<<std::endl;
		vw.write(curr.clone());
    }
	cap.release();
	vw.release();

	Destroy_Algorithm(ptr);

    return 0;
}