/*
  Copyright (c) 2023, Guangfu WANG
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
  * Neither the name of the <organization> nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY <copyright holder> ''AS IS'' AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL <copyright holder> BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <filesystem>
#include <chrono>
#include <unordered_map>

namespace gf
{

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

class Util
{
public:
	static bool checkDirExist(const std::string &dir);
	static bool checkFileExist(const std::string &file);
	static void tic();
	static long toc();
	static long getFileSize(const std::string &file);
	static int cvtStr2Int(const char *arr);

private:
	///note this is thread local variable, enabling the util function can be used in multi-thread environments.
	static thread_local std::chrono::high_resolution_clock::time_point mTic;
};

/**
* @example:
* Factory<BaseClass> f;
* f.registerType<Descendant1>("Descendant1");
* f.registerType<Descendant2>("Descendant2");
* Descendant1* d1 = static_cast<Descendant1*>(f.create("Descendant1"));
* Descendant2* d2 = static_cast<Descendant2*>(f.create("Descendant2"));
* BaseClass *b1 = f.create("Descendant1");
* BaseClass *b2 = f.create("Descendant2");
* @tparam T
*/
template<typename T>
class Factory
{
public:
	template<typename TDerived>
	void registerType(const std::string &name)
	{
		static_assert(std::is_base_of<T, TDerived>::value,
					  "Factory::registerType doesn't accept this type because doesn't derive from base class");
		_createFuncs[name] = &createFunc<TDerived>;
	}

	T *create(const std::string &name)
	{
		typename std::unordered_map<std::string, PCreateFunc>::const_iterator it = _createFuncs.find(name);
		if (it != _createFuncs.end()) {
			return it->second();
		}
		return nullptr;
	}

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
	template<typename TDerived>
	static T *createFunc()
	{
		return new TDerived();
	}

	typedef T *(*PCreateFunc)();

	std::unordered_map<std::string, PCreateFunc> _createFuncs;
};

}