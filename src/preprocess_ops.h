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

#include <yaml-cpp/yaml.h>
#include <opencv4/opencv2/opencv.hpp>
#include "util.h"

namespace gf {
    // Abstraction of preprocessing operation class, copied.
    class PreprocessOp {
    public:
        virtual ~PreprocessOp() = default;

        virtual void Run(cv::cuda::GpuMat *data, const int &num) = 0;

    protected:
        SharedRef<cv::cuda::Stream> m_stream = createSharedRef<cv::cuda::Stream>();
    };

    class NormalizeImage final : public PreprocessOp {
    public:
        void Run(cv::cuda::GpuMat *data, const int &num) override;
    };

    class Permute final : public PreprocessOp {
    public:
        void Run(cv::cuda::GpuMat *data, const int &num) override;
    };

    class Resize final : public PreprocessOp {
    public:
        void Run(cv::cuda::GpuMat *data, const int &num) override;

    private:
        // Compute best resize scale for x-dimension, y-dimension
        static std::pair<float, float> GenerateScale(const cv::cuda::GpuMat &im);
    };

    class LetterBoxResize final : public PreprocessOp {
    public:
        void Run(cv::cuda::GpuMat *data, const int &num) override;

    private:
        static float GenerateScale(const cv::cuda::GpuMat &im);
    };

    // Models with FPN need input shape % stride == 0
    class PadStride final : public PreprocessOp {
    public:
        void Run(cv::cuda::GpuMat *data, const int &num) override;
    };

    class TopDownEvalAffine final : public PreprocessOp {
    public:
        void Run(cv::cuda::GpuMat *data, const int &num) override;
    };
}
