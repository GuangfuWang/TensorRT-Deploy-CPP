#pragma  once

#include <string>
#include <unordered_map>
#include <vector>

namespace gf
{
class TrtResults
{
public:
	virtual ~TrtResults() = default;
	virtual void Get(const std::string &idx_name, std::vector<float> &res) = 0;
	virtual void Set(const std::pair<std::string, std::vector<float>> &data) = 0;
	virtual void Clear() = 0;
protected:
	std::unordered_map<std::string, std::vector<float>> m_res;
};

class FightPPTSMResults final: public TrtResults
{
public:
	void Get(const std::string &idx_name, std::vector<float> &res) override;
	void Set(const std::pair<std::string, std::vector<float>> &data) override;
	void Clear() override;
};
}
