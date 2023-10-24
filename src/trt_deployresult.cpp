#include "trt_deployresult.h"
#include "config.h"

namespace fight
{

void TrtResults::Get(const std::string &idx_name, std::vector<float> &res)
{
	if (std::find(m_config->OUTPUT_NAMES.begin(), m_config->OUTPUT_NAMES.end(),
				  idx_name) != m_config->OUTPUT_NAMES.end()) {
		res = m_res[idx_name];
	}
}

void TrtResults::Set(const std::pair<std::string, std::vector<float>> &data)
{
	if (std::find(m_config->OUTPUT_NAMES.begin(), m_config->OUTPUT_NAMES.end(),
				  data.first) != m_config->OUTPUT_NAMES.end()) {
		m_res[data.first] = data.second;
	}
}

TrtResults::~TrtResults()
{
	m_res.clear();
}
void TrtResults::Clear()
{
	m_res.clear();
}
}

