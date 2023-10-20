#include "config.h"
#include "macro.h"
#include "util.h"

namespace gf {

    thread_local std::string              Config::MODEL_NAME = "../models/ppTSM_fight.engine";

    thread_local std::string              Config::BACKBONE = "ResNet50";

    thread_local std::string              Config::VIDEO_FILE;

    thread_local std::string              Config::RTSP_SITE = "/url/to/rtsp/site";

    thread_local std::vector<int>         Config::INPUT_SHAPE = {1, 8, 3, 320, 320};

    thread_local std::string              Config::INPUT_NAME = "image";

    thread_local std::vector<std::string> Config::OUTPUT_NAMES = {"scores"};

    thread_local unsigned int Config::STRIDE = 2;

    thread_local unsigned int Config::INTERP = 0;

    thread_local unsigned int Config::SAMPLE_INTERVAL = 3;

    thread_local unsigned int Config::TRIGGER_LEN = 8;

    thread_local unsigned int Config::BATCH_SIZE = 1;

    thread_local float Config::THRESHOLD = 0.8f;

    thread_local unsigned int Config::TARGET_CLASS = 1;

    thread_local std::vector<int> Config::TARGET_SIZE = {320, 320};

    thread_local std::vector<int> Config::TRAIN_SIZE = {320, 320};

    thread_local unsigned int Config::SHORT_SIZE = 340;

    thread_local std::vector<std::string> Config::PIPELINE_TYPE = {"TopDownEvalAffine","Resize","LetterBoxResize","NormalizeImage","PadStride","Permute"};

    thread_local std::vector<float>       Config::N_MEAN = {0.485f, 0.456f, 0.406f};

    thread_local std::vector<float>       Config::N_STD = {0.229f, 0.224f, 0.225f};

    thread_local bool Config::ENABLE_SCALE = true;

    thread_local bool Config::KEEP_RATIO = true;

    thread_local bool Config::TIMING = true;

    thread_local int Config::POST_MODE = 0;

    thread_local std::vector<unsigned char> Config::TEXT_COLOR = {255, 0, 0};

    thread_local std::vector<unsigned char> Config::BOX_COLOR = {255, 0, 0};

    thread_local float Config::TEXT_LINE_WIDTH = 2.0f;

    thread_local float Config::BOX_LINE_WIDTH = 2.0;

    thread_local float Config::TEXT_FONT_SIZE = 1.8f;

    thread_local int   Config::TEXT_OFF_X = 450;

    thread_local int   Config::TEXT_OFF_Y = 50;

    thread_local std::string Config::POSTPROCESS_NAME = "FightPpTSMDeployPost";

    thread_local std::string Config::POST_TEXT = "Fight";

	bool Config::init = false;

    void Config::LoadConfigFile(int argc, char **argv, const std::string &file) {
		if(init)return;
		init = true;
        Config::MODEL_NAME = FIGHT_DEPLOY_MODEL;
        Config::INPUT_NAME = FIGHT_INPUT_NAME;
        Config::OUTPUT_NAMES.clear();
        char sep_ = ';';
        const std::string &temp_ = FIGHT_OUTPUT_NAMES;
        if (temp_.find_first_of(sep_) == std::string::npos)sep_ = ',';
        size_t curr = 0;
        size_t lst = 0;
        while (true) {
            curr = temp_.find(sep_, curr);
            Config::OUTPUT_NAMES.emplace_back(temp_.substr(lst, curr - lst));
            if (curr == std::string::npos)break;
            lst = curr;
        }

        if (!Util::checkFileExist(file)) {
            std::cerr << "Config file non exists! Aborting..." << std::endl;
        }

        YAML::Node config;
        config = YAML::LoadFile(file);

        if (config["MODEL"].IsDefined()) {
            auto model_node = config["MODEL"];
            if (model_node["MODEL_NAME"].IsDefined()) {
                Config::MODEL_NAME = model_node["MODEL_NAME"].as<std::string>();
            }
            if (model_node["BACKBONE"].IsDefined()) {
                Config::BACKBONE = model_node["BACKBONE"].as<std::string>();
            }
            if (model_node["INPUT_NAME"].IsDefined()) {
                Config::INPUT_NAME = model_node["INPUT_NAME"].as<std::string>();
            }
            if (model_node["OUTPUT_NAMES"].IsDefined()) {
                Config::OUTPUT_NAMES = model_node["OUTPUT_NAMES"].as<std::vector<std::string>>();
            }
        } else {
            std::cerr << "Please set MODEL, " << std::endl;
        }

        if (config["DATA"].IsDefined()) {
            auto model_node = config["DATA"];
            if (model_node["VIDEO_NAME"].IsDefined()) {
                Config::VIDEO_FILE = model_node["VIDEO_NAME"].as<std::string>();
            }
            if (model_node["RTSP_SITE"].IsDefined()) {
                Config::RTSP_SITE = model_node["RTSP_SITE"].as<std::string>();
            }
            if (model_node["INPUT_SHAPE"].IsDefined()) {
                Config::INPUT_SHAPE = model_node["INPUT_SHAPE"].as<std::vector<int>>();
            }
        } else {
            std::cerr << "Please set DATA, " << std::endl;
        }

        if (config["PIPELINE"].IsDefined()) {
            auto model_node = config["PIPELINE"];
            if (model_node["STRIDE"].IsDefined()) {
                Config::STRIDE = model_node["STRIDE"].as<unsigned int>();
            }
            if (model_node["INTERP"].IsDefined()) {
                Config::INTERP = model_node["INTERP"].as<unsigned int>();
            }
            if (model_node["SAMPLE_INTERVAL"].IsDefined()) {
                Config::SAMPLE_INTERVAL = model_node["SAMPLE_INTERVAL"].as<unsigned int>();
            }
            if (model_node["TRIGGER_LEN"].IsDefined()) {
                Config::TRIGGER_LEN = model_node["TRIGGER_LEN"].as<unsigned int>();
            }
            if (model_node["BATCH_SIZE"].IsDefined()) {
                Config::BATCH_SIZE = model_node["BATCH_SIZE"].as<unsigned int>();
            }
            if (model_node["THRESHOLD"].IsDefined()) {
                Config::THRESHOLD = model_node["THRESHOLD"].as<float>();
            }
            if (model_node["TARGET_CLASS"].IsDefined()) {
                Config::TARGET_CLASS = model_node["TARGET_CLASS"].as<unsigned int>();
            }
            if (model_node["ENABLE_SCALE"].IsDefined()) {
                Config::ENABLE_SCALE = model_node["ENABLE_SCALE"].as<bool>();
            }
            if (model_node["KEEP_RATIO"].IsDefined()) {
                Config::KEEP_RATIO = model_node["KEEP_RATIO"].as<bool>();
            }
            if (model_node["TIMING"].IsDefined()) {
                Config::TIMING = model_node["TIMING"].as<bool>();
            }
            if (model_node["TARGET_SIZE"].IsDefined()) {
                Config::TARGET_SIZE = model_node["TARGET_SIZE"].as<std::vector<int>>();
            }
            if (model_node["TRAIN_SIZE"].IsDefined()) {
                Config::TRAIN_SIZE = model_node["TRAIN_SIZE"].as<std::vector<int>>();
            }
            if (model_node["SHORT_SIZE"].IsDefined()) {
                Config::SHORT_SIZE = model_node["SHORT_SIZE"].as<unsigned int>();
            }
            if (model_node["PIPELINE_TYPE"].IsDefined()) {
                Config::PIPELINE_TYPE = model_node["PIPELINE_TYPE"].as<std::vector<std::string>>();
            }
            if (model_node["N_MEAN"].IsDefined()) {
                Config::N_MEAN = model_node["N_MEAN"].as<std::vector<float>>();
            }
            if (model_node["N_STD"].IsDefined()) {
                Config::N_STD = model_node["N_STD"].as<std::vector<float>>();
            }

        } else {
            std::cerr << "Please set PIPELINE, " << std::endl;
        }

        if (config["POSTPROCESS"].IsDefined()) {
            auto model_node = config["POSTPROCESS"];
            if (model_node["POST_MODE"].IsDefined()) {
                Config::POST_MODE = model_node["POST_MODE"].as<int>();
            }
            if (model_node["TEXT_COLOR"].IsDefined()) {
                Config::TEXT_COLOR = model_node["TEXT_COLOR"].as<std::vector<unsigned char>>();
                std::swap(Config::TEXT_COLOR[0], Config::TEXT_COLOR[2]);
            }
            if (model_node["BOX_COLOR"].IsDefined()) {
                Config::BOX_COLOR = model_node["BOX_COLOR"].as<std::vector<unsigned char>>();
                std::swap(Config::BOX_COLOR[0], Config::BOX_COLOR[2]);
            }
            if (model_node["TEXT_LINE_WIDTH"].IsDefined()) {
                Config::TEXT_LINE_WIDTH = model_node["TEXT_LINE_WIDTH"].as<float>();
            }
            if (model_node["BOX_LINE_WIDTH"].IsDefined()) {
                Config::BOX_LINE_WIDTH = model_node["BOX_LINE_WIDTH"].as<float>();
            }
            if (model_node["TEXT_FONT_SIZE"].IsDefined()) {
                Config::TEXT_FONT_SIZE = model_node["TEXT_FONT_SIZE"].as<float>();
            }
            if (model_node["TEXT_OFF_X"].IsDefined()) {
                Config::TEXT_OFF_X = model_node["TEXT_OFF_X"].as<int>();
                if (Config::TEXT_OFF_X < 0) {
                    Config::TEXT_OFF_X = Config::INPUT_SHAPE.back() / 2 - 5;
                }
            }
            if (model_node["TEXT_OFF_Y"].IsDefined()) {
                Config::TEXT_OFF_Y = model_node["TEXT_OFF_Y"].as<int>();
            }
            if (model_node["POSTPROCESS_NAME"].IsDefined()) {
                Config::POSTPROCESS_NAME = model_node["POSTPROCESS_NAME"].as<std::string>();
            }
            if (model_node["POST_TEXT"].IsDefined()) {
                Config::POST_TEXT = model_node["POST_TEXT"].as<std::string>();
            }
        } else {
            std::cerr << "Please set MODEL, " << std::endl;
        }
		if(argc<2)return;
        cmdline::parser parser;
        parser.add<std::string>("input_name", 'i', "Input layer name for trt.", false);
        parser.add<std::string>("output_names", 'o', "Output layer names for trt.", false);
        parser.add<std::string>("model_name", 'm', "Model name for trt.", false);
        parser.add<std::string>("video_file", 'v', "Video file for trt.", false);
        parser.parse_check(argc, argv);

        std::string InLayerName = parser.get<std::string>("input_name");
        std::string OutLayerNames = parser.get<std::string>("output_names");
        std::string ModelName = parser.get<std::string>("model_name");
        std::string VideoFile = parser.get<std::string>("video_file");

        if (!InLayerName.empty()) {
            Config::INPUT_NAME = InLayerName;
        }
        if (!OutLayerNames.empty()) {
            Config::OUTPUT_NAMES.clear();
            size_t cur = 0;
            char sep = ';';
            const std::string &temp = OutLayerNames;
            if (temp.find_first_of(sep) == std::string::npos)sep = ',';
            cur = 0;
            size_t last = 0;
            while (true) {
                cur = temp.find(sep, cur);
                Config::OUTPUT_NAMES.emplace_back(temp.substr(last, cur - last));
                if (cur == std::string::npos)break;
                last = cur;
            }
        }
        if (!ModelName.empty()) {
            Config::MODEL_NAME = ModelName;
        }

        if (!gf::Util::checkFileExist(Config::MODEL_NAME)) {
            std::cout << Config::MODEL_NAME << std::endl;
            std::cerr << "Model does not exists!" << std::endl;
            std::cerr << "Please check the model path..." << std::endl;
        }

        if (!VideoFile.empty()) {
            Config::VIDEO_FILE = VideoFile;
        }
    }

}

