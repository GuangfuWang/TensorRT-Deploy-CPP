#include <opencv2/imgproc.hpp>
#include "postprocessor.h"
#include "config.h"

namespace gf {

    thread_local bool Postprocessor::INIT_FLAG = false;

    void FightPpTSMDeployPost::Run(const SharedRef<TrtResults> &res, const std::vector<cv::Mat> &img,
                                   std::vector<cv::Mat> &out_img) {
        //our simple program will only draw letters on top of images.
        auto flag = static_cast<PostProcessFlag>(Config::POST_MODE);
        assert(flag == PostProcessFlag::DRAW_LETTER);
        std::vector<float> fight_res;
        res->Get(Config::OUTPUT_NAMES[0], fight_res);
        Util::softmax(fight_res);

        m_moving_average.push_back(fight_res[Config::TARGET_CLASS]);
        float sum=0.0f;
        if (m_moving_average.size()>3){
            for(auto i = m_moving_average.size()-1;i>=m_moving_average.size()-3;--i){
                sum+=m_moving_average[i];
            }
            sum/=3.0f;
        }
        std::stringstream text;
        text << Config::POST_TEXT << ": " << 100 * sum << "%";
        std::cout<<text.str()<<std::endl;
        for (int i = 0; i < img.size(); ++i) {
            ///@note the putText method does not have GPU version since it quite slow running on GPU for per pixel ops.
            if (sum>=Config::THRESHOLD){
                for (int j = 0; j < Config::SAMPLE_INTERVAL; ++j) {
                    cv::putText(out_img[i*Config::SAMPLE_INTERVAL+j], text.str(),
                                cv::Point(Config::TEXT_OFF_X, Config::TEXT_OFF_Y),
                                cv::FONT_HERSHEY_PLAIN, Config::TEXT_FONT_SIZE,
                                cv::Scalar(Config::TEXT_COLOR[0], Config::TEXT_COLOR[1], Config::TEXT_COLOR[2]),
                                (int) Config::TEXT_LINE_WIDTH);
                }

            }

        }

    }

    void Postprocessor::Init() {
        if (!m_ops) {
            m_ops = createSharedRef<Factory<PostprocessorOps>>();
        }
        m_ops->registerType<FightPpTSMDeployPost>(Config::POSTPROCESS_NAME);
        m_worker = m_ops->create(Config::POSTPROCESS_NAME);
    }

    void Postprocessor::Run(const SharedRef<TrtResults> &res, const std::vector<cv::Mat> &img,
                            std::vector<cv::Mat> &out_img) {
        if (!INIT_FLAG) {
            Init();
            INIT_FLAG = true;
        }
        m_worker->Run(res, img, out_img);
    }

    Postprocessor::~Postprocessor() {
        if (m_ops) {
            m_ops->destroy();
        }
    }

}