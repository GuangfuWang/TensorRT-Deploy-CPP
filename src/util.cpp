#include "util.h"
#include <sys/stat.h>
#include <iostream>
#include <valarray>

namespace gf
{

thread_local std::chrono::high_resolution_clock::time_point Util::mTic;

bool Util::checkDirExist(const std::string &dir)
{
	///note this feature require c++ 17 and above.
	return std::filesystem::exists(dir);
}

bool Util::checkFileExist(const std::string &file)
{
	///note this feature require c++ 17 and above.
	return std::filesystem::exists(file);
}

void Util::tic()
{
	mTic = std::chrono::high_resolution_clock::now();
}

long Util::toc()
{
	auto dur = std::chrono::high_resolution_clock::now() - mTic;
	mTic = std::chrono::high_resolution_clock::now();
	long ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	return ms;
}

long Util::getFileSize(const std::string &file)
{
	if (!checkFileExist(file)) {
		std::cerr << "File: " << file << " not exists..." << std::endl;
		return -1;
	}
	struct stat statBuf;
	stat(file.c_str(), &statBuf);
	return statBuf.st_size;
}

int Util::cvtStr2Int(const char *arr)
{
	return std::atoi(arr);
}
void Util::softmax(std::vector<float> &in)
{
	float total = 0.0f;
	for (auto &e : in) {
		auto curr = std::exp(e);
		e = curr;
		total += curr;
	}
	for (auto &e : in) {
		e /= total;
	}
}
}
