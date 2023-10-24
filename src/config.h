#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "cmdline.h"

namespace fight
{
/**
 * @brief config class for deployment. should not be derived.
 */
class Config final
{
public:
	explicit Config(int argc, char **argv, const std::string &file = "../config/fight_detection.yaml");
	/**
	 * @brief loading the config yaml file, default folder is ./weight/fight/fight_detection.yaml
	 * @param argc terminal arg number.
	 * @param argv terminal arg values.
	 * @param file config file full path.
	 * @note priority: 1 Terminal; 2 Config file; 3 Compilation settings
	 */
	void LoadConfigFile(int argc, char **argv, const std::string &file);
public:
	std::string MODEL_NAME = "../models/ppTSM_fight.engine";
	std::string BACKBONE = "ResNet50";
	std::string VIDEO_FILE;
	std::string RTSP_SITE = "/url/to/rtsp/site";
	std::vector<int> INPUT_SHAPE = {1, 8, 3, 320, 320};
	std::string INPUT_NAME = "image";
	std::vector<std::string> OUTPUT_NAMES = {"scores"};
	unsigned int STRIDE = 2;
	unsigned int INTERP = 0;
	unsigned int SAMPLE_INTERVAL = 3;
	unsigned int TRIGGER_LEN = 8;
	unsigned int BATCH_SIZE = 1;
	float THRESHOLD = 0.8f;
	unsigned int TARGET_CLASS = 1;
	std::vector<int> TARGET_SIZE = {320, 320};
	std::vector<int> TRAIN_SIZE = {320, 320};
	unsigned int SHORT_SIZE = 340;
	std::vector<std::string> PIPELINE_TYPE = {"TopDownEvalAffine","Resize",
											  "LetterBoxResize","NormalizeImage"};

	std::vector<float> N_MEAN = {0.485f, 0.456f, 0.406f};
	std::vector<float> N_STD = {0.229f, 0.224f, 0.225f};
	bool ENABLE_SCALE = true;
	bool KEEP_RATIO = true;
	bool TIMING = true;
	int POST_MODE = 0;
	std::vector<unsigned char> TEXT_COLOR = {255, 0, 0};
	std::vector<unsigned char> BOX_COLOR = {255, 0, 0};
	float TEXT_LINE_WIDTH = 30.0f;
	float BOX_LINE_WIDTH = 2.0;
	float TEXT_FONT_SIZE = 1.8f;
	int   TEXT_OFF_X = 450;
	int   TEXT_OFF_Y = 50;
	std::string POSTPROCESS_NAME = "FightPpTSMDeployPost";
	std::string POST_TEXT = "Fight";
	std::string POST_TEXT_FONT_FILE = "";
	bool init = false;
};
}
