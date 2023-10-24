#include "model.h"
#include "config.h"
#include "trt_deploy.h"
#include "trt_deployresult.h"

namespace fight {



class InferModel {
public:
    explicit InferModel(SharedRef<Config>& config) {
		m_config = config;
        mDeploy = createSharedRef<TrtDeploy>(config);
        mResult = createSharedRef<TrtResults>(config);
    }

public:
    SharedRef<TrtDeploy> mDeploy;
    SharedRef<TrtResults> mResult;
    std::vector<cv::Mat> mSampled;
	std::vector<cv::Mat> mTotal;
	SharedRef<Config> m_config;
	unsigned int COUNT_LOOP = 0;
	unsigned int COUNT = 0;
	unsigned int COUNT_TOTAL = 0;
};

void *GenModel(SharedRef<Config>& config) {
    auto *model = new InferModel(config);
    return reinterpret_cast<void *>(model);
}

cv::Mat genROI(const cv::Mat &im, const std::vector<int> &points, const cv_Point *coords)
{
	cv::Mat roi_img = cv::Mat::zeros(im.size(),CV_8UC3);
	if(points.empty())return cv::Mat(im.size(),CV_8UC3,cv::Scalar(255,255,255));
	std::vector<std::vector<cv::Point>> contour;

	int sums = 0;
	for(auto& each:points){
		std::vector<cv::Point> pts;
		for (int j = sums; j < each+sums; ++j) {
			pts.push_back(cv::Point(coords[j].x, coords[j].y));
		}
		sums+=each;
		contour.push_back(pts);
	}
	sums = 0;
	for(auto& i:points){
		cv::drawContours(roi_img, contour, sums,
						 cv::Scalar::all(255), -1);
		sums++;
	}
	return roi_img;
}

void plotLines(cv::Mat &im, const std::vector<int> &points,
					 const cv_Point * coords,const int& thickness)
{
	int sums = 0;
	for(auto& each:points){
		for (int j = sums; j < each+sums; ++j) {
			int k = j+1;
			if(k==each+sums)k=sums;
			cv::line(im, cv::Point(coords[j].x, coords[j].y),
					 cv::Point(coords[k].x, coords[k].y), cv::Scalar(255, 0, 0),
					 thickness);
		}
		sums+=each;
	}
}

cvModel *Allocate_Algorithm(cv::Mat &input_frame, int algID, int gpuID) {
	cv::cuda::setDevice(gpuID);
	cudaSetDevice(gpuID);
    std::string file;
    if(Util::checkFileExist("./fight_detection.yaml"))
        file = "./fight_detection.yaml";
    else if(Util::checkFileExist("../config/fight_detection.yaml")){
        file = "../config/fight_detection.yaml";
    }else{
        std::cout<<"Cannot find YAML file!"<<std::endl;
    }
    auto config = createSharedRef<Config>(0, nullptr,file);
    auto *ptr = new cvModel();
    ptr->FrameNum = 0;
    ptr->Frameinterval = 0;
    ptr->countNum = 0;
    ptr->width = input_frame.cols;
    ptr->height = input_frame.rows;
    ptr->iModel = GenModel(config);
	auto model = reinterpret_cast<InferModel *>(ptr->iModel);
    model->COUNT = config->SAMPLE_INTERVAL*config->TRIGGER_LEN;
    return ptr;
}

void SetPara_Algorithm(cvModel *pModel, int algID) {
    //todo: implement this
}

void UpdateParams_Algorithm(cvModel *pModel) {
    //todo: implement this
}

void Process_Algorithm(cvModel *pModel, cv::Mat &input_frame) {
    pModel->alarm = 0;
	auto model = reinterpret_cast<InferModel *>(pModel->iModel);
	auto config = model->m_config;

	auto roi = pModel->p;
	cv::Mat roi_img = genROI(input_frame,pModel->pointNum,roi);
	cv::Mat removed_roi;
	input_frame.copyTo(removed_roi,roi_img);

    if (model->COUNT_LOOP < model->COUNT) {
        if((model->COUNT_TOTAL%config->SAMPLE_INTERVAL)==0){
            model->mSampled.push_back(removed_roi.clone());

        }
		if(config->POST_MODE!=4)model->mTotal.push_back(input_frame.clone());
		model->COUNT_LOOP++;
    }else{
        model->mDeploy->Infer(model->mSampled, model->mResult);
		model->mDeploy->Postprocessing(model->mResult,model->mSampled,model->mTotal,pModel->alarm);
		if(pModel->alarm)input_frame = model->mTotal[model->mTotal.size()-1];
		model->mSampled.clear();
        model->mSampled.push_back(removed_roi.clone());
		model->COUNT_LOOP  = 1;
		if(config->POST_MODE!=4){
			model->mTotal.clear();
			model->mTotal.push_back(input_frame.clone());
		}
    }
	model->COUNT_TOTAL++;
	plotLines(input_frame,pModel->pointNum,
							   roi,(int)config->BOX_LINE_WIDTH);
}

void Destroy_Algorithm(cvModel *pModel) {
    if (pModel->iModel){
        auto model = reinterpret_cast<InferModel *>(pModel->iModel);
        delete model;
        model = nullptr;
    }
    if (pModel) {
        delete pModel;
        pModel = nullptr;
    }
}

} // namespace fight
