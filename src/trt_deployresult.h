#pragma  once

#include <string>
#include <unordered_map>
#include <vector>
#include "config.h"
#include "util.h"

namespace fight
{
/**
 * @brief This is a class used for storing inference results.
 * @details This is a base class, and can be used inside a TrtDeploy class, <!--
 * --> to be more general, this class is designed to be pure virtual class and the Get() & Set() methods <!--
 * --> should be implemented.
 * @example:
 * @code
 * 	SharedRef<TrtDeploy> mDeploy = createSharedRef<TrtDeploy>();
 *	SharedRef<TrtResults> mResult = createSharedRef<TrtResults>();
 *	mDeploy->Warmup(mResult);
 *	std::vector<float> score;
 *  mResults->Get("scores",score);
 *  std::cout<<score<<std::endl;
 * @endcode
 *
 */
class TrtResults final
{
public:
	explicit TrtResults(SharedRef<Config>& config):m_config(config){};
	/**
	 * @brief default de-constructor, to empty the map.
	 */
	~TrtResults();
	/**
	 * @brief interface function, to get data by name.
	 * @param idx_name index name, i.e. "scores" for binary classification problem.
	 * @param res result data copied from GPU, stored as float vector.
	 */
	void Get(const std::string &idx_name, std::vector<float> &res);
	/**
	 * @brief interface function, mostly used inside the TrtDeploy::Infer() method.
	 * @param data paired data, copied from GPU using cuda.
	 */
	void Set(const std::pair<std::string, std::vector<float>> &data);
	/**
	 * @brief clear the map.
	 */
	void Clear();
private:
	std::unordered_map<std::string, std::vector<float>> m_res;///< map for storing the current inference data.
	SharedRef<Config> m_config = nullptr;
};

}
