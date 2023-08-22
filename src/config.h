#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "cmdline.h"

namespace gf
{
class Config final
{
public:
	///@note priority: 1 Terminal; 2 Config file; 3 Compilation settings
	static void LoadConfigFile(int argc, char **argv, const std::string &file);
public:
	static thread_local std::string              MODEL_NAME;
	static thread_local std::string              BACKBONE;
	static thread_local std::string              VIDEO_FILE;
	static thread_local std::string              RTSP_SITE;
	static thread_local std::vector<int>         INPUT_SHAPE;
	static thread_local std::string              INPUT_NAME;
	static thread_local std::vector<std::string> OUTPUT_NAMES;
	static thread_local unsigned int             STRIDE;
	static thread_local unsigned int             INTERP;
	static thread_local unsigned int             SAMPLE_INTERVAL;
	static thread_local unsigned int             TRIGGER_LEN;
	static thread_local unsigned int             BATCH_SIZE;
	static thread_local float                    THRESHOLD;
	static thread_local std::vector<int>         TARGET_SIZE;
	static thread_local std::vector<int>         TRAIN_SIZE;
	static thread_local unsigned int             SHORT_SIZE;
	static thread_local std::vector<std::string> PIPELINE_TYPE;
	static thread_local std::vector<float>       N_MEAN;
	static thread_local std::vector<float>       N_STD;
	static thread_local bool                     ENABLE_SCALE;
	static thread_local bool                     KEEP_RATIO;
	static thread_local bool                     TIMING;

	static thread_local int                        POST_MODE;
	static thread_local std::vector<unsigned char> TEXT_COLOR;
	static thread_local std::vector<unsigned char> BOX_COLOR;
	static thread_local float                      TEXT_LINE_WIDTH;
	static thread_local float                      BOX_LINE_WIDTH;
	static thread_local float                      TEXT_FONT_SIZE;
	static thread_local int                        TEXT_OFF_X;
	static thread_local int                        TEXT_OFF_Y;
	static thread_local std::string                POSTPROCESS_NAME;
};
}