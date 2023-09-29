#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "cmdline.h"

namespace gf
{
/**
 * @brief config class for deployment. should not be derived.
 */
class Config final
{
public:
	/**
	 * @brief loading the config yaml file, default folder is ./weight/fight/infer_cfg.yaml
	 * @param argc terminal arg number.
	 * @param argv terminal arg values.
	 * @param file config file full path.
	 * @note priority: 1 Terminal; 2 Config file; 3 Compilation settings
	 */
	static void LoadConfigFile(int argc, char **argv, const std::string &file);
public:
	static thread_local std::string MODEL_NAME;///< name of trained model,should be tensorrt model.
	static thread_local std::string BACKBONE;///< backbone of current model.
	static thread_local std::string VIDEO_FILE;///< video file path.
	static thread_local std::string RTSP_SITE;///< online RTSP URL.
	static thread_local std::vector<int> INPUT_SHAPE;///< input data shape.
	static thread_local std::string INPUT_NAME;///< input name inside the model.
	static thread_local std::vector<std::string> OUTPUT_NAMES;///< output names inside the model, possibly multi-head.
	static thread_local unsigned int STRIDE;///< stride for padding.
	static thread_local unsigned int INTERP;///< interpolation for opencv resizing.
	static thread_local unsigned int SAMPLE_INTERVAL;///< under which we will sample an image from video or streams.
	static thread_local unsigned int TRIGGER_LEN;///< until which we will conduct an inference.
	static thread_local unsigned int BATCH_SIZE;///< currently must be 1.
	static thread_local float THRESHOLD;///< threshold for binary classification.
	static thread_local unsigned int TARGET_CLASS;///< indicate which class to use.
	static thread_local std::vector<int> TARGET_SIZE;///< target images size.
	static thread_local std::vector<int> TRAIN_SIZE;///< input tensor size for model.
	static thread_local unsigned int SHORT_SIZE;///< not unknown.
	static thread_local std::vector<std::string> PIPELINE_TYPE;///< preprocessing pipeline names and orders.
	static thread_local std::vector<float> N_MEAN;///< channel wise normalization mean.
	static thread_local std::vector<float> N_STD;///< channel wise normalization standard deviation.
	static thread_local bool ENABLE_SCALE;///< enable scale of images.
	static thread_local bool KEEP_RATIO;///< keep the height/width ratio when scale.
	static thread_local bool TIMING;///< enable timing of FPS.

	static thread_local int POST_MODE;///< post processing mode, Current 4 types of mode are supported: DRAW_LETTER = 0, DRAW_BOX = 1, DRAW_BOX_LETTER = 2,MASK_OUT = 3
	static thread_local std::vector<unsigned char> TEXT_COLOR;///< text color for draw text into images, should be R,G,B
	static thread_local std::vector<unsigned char> BOX_COLOR;///< box color for draw box, should be R,G,B. i.e.[255,0,0]
	static thread_local float TEXT_LINE_WIDTH;///< text line width for drawing.
	static thread_local float BOX_LINE_WIDTH;///< box line width for drawing.
	static thread_local float TEXT_FONT_SIZE;///<text font size for drawing.
	static thread_local int TEXT_OFF_X;///< text drawing position offset x.
	static thread_local int TEXT_OFF_Y;///< text drawing position offset y.
	static thread_local std::string POSTPROCESS_NAME;///< post processor name, should be same as used. (subclass of PostprocessorOps)
	static thread_local std::string POST_TEXT;///< post processing text.
};
}