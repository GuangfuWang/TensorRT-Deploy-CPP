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

int main(int argc, char **argv) {
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
    std::vector<std::vector<cv::Mat>> imgs;
//    imgs.resize(total / Config::TRIGGER_LEN);
//    for (int i = 0; i < imgs.size(); ++i) {
//        imgs[i].resize(Config::TRIGGER_LEN);
//    }
//    std::vector<cv::Mat> out_data;
//    cv::Mat img;
//    int jdx = 0;
//    while (cap.read(img)) {
//        imgs[jdx / Config::TRIGGER_LEN][jdx % Config::TRIGGER_LEN] = img.clone();
//        jdx++;
//        if (jdx / Config::TRIGGER_LEN == imgs.size())break;
//    }
//
//    for (auto &each: imgs) {
//        mDeploy->Infer(each, mResult);
//        mDeploy->Postprocessing(mResult, each, out_data);
//        for (auto &f: out_data) {
//            vw.write(f);
//        }
//        out_data.clear();
//    }
    cv::Mat img;
    int c = 0;
    std::vector<std::vector<cv::Mat>> unsampled;
    std::vector<cv::Mat> cur;
    std::vector<cv::Mat> whole;
    int num = total / (Config::TRIGGER_LEN * Config::SAMPLE_INTERVAL);
    while (cap.read(img)) {
        if (c / (Config::TRIGGER_LEN * Config::SAMPLE_INTERVAL) == num)break;
        if (c % Config::SAMPLE_INTERVAL == 0) {
            cur.emplace_back(img.clone());
        }
        whole.emplace_back(img.clone());
        c++;
        if (cur.size() == Config::TRIGGER_LEN) {
            imgs.emplace_back(cur);
            unsampled.emplace_back(whole);
            cur.clear();
            whole.clear();
        }
    }
    c = 0;

    for (auto &each: imgs) {
        mDeploy->Infer(each, mResult);
        mDeploy->Postprocessing(mResult, each, unsampled[c]);
        c++;
    }
    for (auto &f: unsampled) {
        for (auto & single:f) {
            vw.write(single);
        }
    }
    unsampled.clear();

    return 0;
}