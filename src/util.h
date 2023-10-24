#pragma once

#include <filesystem>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>

namespace fight
{
/**
 * @brief this is used for lazy purpose.
 */
template<typename T>
using UniqueRef = std::unique_ptr<T>;
template<typename T>
using SharedRef = std::shared_ptr<T>;

///@note here is cpp perfect forwarding.
template<typename T, typename ... Args>
constexpr SharedRef<T> createSharedRef(Args &&... args)
{
	return std::make_shared<T>(std::forward<Args>(args)...);
}

template<typename T, typename ... Args>
constexpr UniqueRef<T> createUniqueRef(Args &&... args)
{
	return std::make_unique<T>(std::forward<Args>(args)...);
}

/**
 * @brief utility class for file operation and timing.
 * @note timing should only be used without jumping to another timing.
 * @warning this class requires C++ 17 standard.
 */
class Util
{
public:
	/**
	 * @brief check a directory exists or not.
	 * @param dir query directory.
	 * @return true if specified directory exists, false if not.
	 */
	static bool checkDirExist(const std::string &dir);
	/**
 	* @brief check a file exists or not.
 	* @param dir query file.
 	* @return true if specified file exists, false if not.
 	*/
	static bool checkFileExist(const std::string &file);
	/**
	 * @brief timing starts.
	 */
	static void tic();
	/**
	 * @brief timing ends.
	 * @return total time in ms since last tic().
	 */
	static long toc();
	/**
	 * @brief get a file size in byte without open it.
	 * @param file query file name.
	 * @return file size in byte.
	 */
	static long getFileSize(const std::string &file);
	/**
	 * @brief converting a string into integer.
	 * @param arr string literal
	 * @return integer.
	 */
	static int cvtStr2Int(const char *arr);

	/**
	 * @brief calculate the softmax.
	 * @param in
	 */
	static void softmax(std::vector<float>& in);

	static std::vector<std::string> parseNames(const std::string &names, char delim);


private:
	///note this is thread local variable, enabling the util function can be used in multi-thread environments.
	static thread_local std::chrono::high_resolution_clock::time_point mTic;///< timer start.
};

/**
 * @brief This is a factory class used as a design pattern.
 * @details This class can serve as a simple implementation of factory pattern, one can register a derived class with given names, and afterwards the registered class can be initialized.
 * @warning This class may not be used in multi-thread environment since by default it will create static objects.
 * @see https://www.geeksforgeeks.org/factory-method-for-designing-pattern/
 *
 *
 * @code
 * Factory<BaseClass> f;
 * f.registerType<Descendant1>("Descendant1");
 * f.registerType<Descendant2>("Descendant2");
 * Descendant1* d1 = static_cast<Descendant1*>(f.create("Descendant1"));
 * Descendant2* d2 = static_cast<Descendant2*>(f.create("Descendant2"));
 * BaseClass *b1 = f.create("Descendant1");
 * BaseClass *b2 = f.create("Descendant2");
 * f.destroy();
 * @endcode
 * @tparam T Base class
 *
*/
template<typename T>
class Factory
{
public:
	/**
	 * @brief this is the register function used for register a target subclass.
	 * @note the subclass must be a sub class of T.
	 * @tparam TDerived derived class name.
	 * @param name registered class name.
	 */
	template<typename TDerived>
	void registerType(const std::string &name)
	{
		static_assert(std::is_base_of<T, TDerived>::value,
					  "Factory::registerType doesn't accept this type because doesn't derive from base class");
		_createFuncs[name] = &createFunc<TDerived>;
	}

	/**
	 * @brief get a static object.
	 * @param name index object by string.
	 * @return pointer to base class.
	 */
	T *create(const std::string &name)
	{
		typename std::unordered_map<std::string, PCreateFunc>::const_iterator it = _createFuncs.find(name);
		if (it != _createFuncs.end()) {
			return it->second();
		}
		return nullptr;
	}

	/**
	 * @brief destroy all static objects.
	 */
	void destroy()
	{
		for (auto &[name, ops] : _createFuncs) {
			typename std::unordered_map<std::string, PCreateFunc>::const_iterator it = _createFuncs.find(name);
			if (it != _createFuncs.end()) {
				delete it->second();
			}
		}
	}

private:
	/**
	 * @brief actual creation of objects.
	 * @tparam TDerived
	 * @return
	 */
	template<typename TDerived>
	static T *createFunc()
	{
		return new TDerived();
	}

	typedef T *(*PCreateFunc)();///< function pointer, can be substitute by std::function.

	std::unordered_map<std::string, PCreateFunc> _createFuncs;///< map stores all objects.
};

}