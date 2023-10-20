#include "model.h"
#include "config.h"
#include "trt_deploy.h"
#include "trt_deployresult.h"

namespace gf {

float CURRENT_RES = 0.0f;
unsigned int COUNT_TOTAL = 0;
unsigned int COUNT_LOOP = 0;
unsigned int COUNT = 0;

class InferModel {
public:
    InferModel() {
        mDeploy = createSharedRef<TrtDeploy>();
        mResult = createSharedRef<TrtResults>();
    }

public:
    SharedRef<TrtDeploy> mDeploy;
    SharedRef<TrtResults> mResult;
    std::vector<cv::Mat> mSampled;
};

static void *GenModel() {
    InferModel *model = new InferModel();
    return reinterpret_cast<void *>(model);
}

static void Plot(cv::Mat &in){
    if (CURRENT_RES>=Config::THRESHOLD){
        std::stringstream text;
        text << Config::POST_TEXT << ": " << 100 * CURRENT_RES << "%";
        cv::putText(in, text.str(),
                    cv::Point(Config::TEXT_OFF_X, Config::TEXT_OFF_Y),
                    cv::FONT_HERSHEY_PLAIN, Config::TEXT_FONT_SIZE,
                    cv::Scalar(Config::TEXT_COLOR[0], Config::TEXT_COLOR[1], Config::TEXT_COLOR[2]),
                    (int) Config::TEXT_LINE_WIDTH);
    }
}

cvModel *Allocate_Algorithm(cv::Mat &input_frame, int algID, int gpuID) {
    std::string file;
    if(Util::checkFileExist("./fight_detection.yaml"))
        file = "./fight_detection.yaml";
    else if(Util::checkFileExist("weight/fight/fight_detection.yaml")){
        file = "weight/fight/fight_detection.yaml";
    }else{
        std::cout<<"Cannot find YAML file!"<<std::endl;
    }
    Config::LoadConfigFile(0, nullptr, file);
    auto *ptr = new cvModel();
    ptr->FrameNum = 0;
    ptr->Frameinterval = 0;
    ptr->countNum = 0;
    ptr->width = input_frame.cols;
    ptr->height = input_frame.rows;
//    Config::INPUT_SHAPE[4] = ptr->width;
//    Config::INPUT_SHAPE[3] = ptr->height;
    ptr->scaleX = (Config::TARGET_SIZE[1]) / (ptr->width);
    ptr->scaleY = (Config::TARGET_SIZE[0]) / (ptr->height);
    ptr->iModel = GenModel();
    COUNT = Config::SAMPLE_INTERVAL*Config::TRIGGER_LEN;
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
    if (COUNT_LOOP < COUNT) {
        auto model = reinterpret_cast<InferModel *>(pModel->iModel);
        if((COUNT_TOTAL%Config::SAMPLE_INTERVAL)==0){
            model->mSampled.emplace_back(input_frame.clone());
        }
        COUNT_LOOP++;
        Plot(input_frame);
    }else{
        auto model = reinterpret_cast<InferModel *>(pModel->iModel);
        model->mDeploy->Infer(model->mSampled, model->mResult);
        std::vector<float> fight_res;
        model->mResult->Get(Config::OUTPUT_NAMES[0], fight_res);
        Util::softmax(fight_res);
        CURRENT_RES = fight_res[Config::TARGET_CLASS];
        Plot(input_frame);
        model->mSampled.clear();
        model->mSampled.push_back(input_frame.clone());
        COUNT_LOOP = 1;
        pModel->alarm = CURRENT_RES>=Config::THRESHOLD;
    }
    COUNT_TOTAL++;
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

} // namespace gf