#include <fstream>
#include <iostream>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/cudaarithm.hpp>
#include "trt_deploy.h"
#include "util.h"

namespace fight {

    void Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
        if (severity != Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }

    TrtDeploy::TrtDeploy(SharedRef<Config>& config) {
		m_config = config;
        m_model_load_status = ModelLoadStatus::NON_LOADED;
        m_cuda_alloc_status = CudaMemAllocStatus::NON_ALLOC;
        m_curr_fps = 0.0f;
    }

    TrtDeploy::~TrtDeploy() {
        cudaStreamSynchronize(m_stream);
        for (int i = 0; i <= m_config->OUTPUT_NAMES.size(); ++i) {
            cudaFree(m_device_ptr[i]);
        }
        cudaStreamDestroy(m_stream);

        if (m_execution_context) {
            delete m_execution_context;
            m_execution_context = nullptr;
        }
        if (m_engine) {
            delete m_engine;
            m_engine = nullptr;
        }
        if (m_runtime) {
            delete m_runtime;
            m_runtime = nullptr;
        }
        std::cout << "TensorRT Deploy Backend Deconstructed..." << std::endl;
    }

    void TrtDeploy::Infer(const std::vector<cv::Mat> &img, SharedRef<TrtResults> &result) {
//        if (m_config->TIMING) {
//            Util::tic();
//        }
        if (!INIT_FLAG) {
            Init(m_config->MODEL_NAME);
            INIT_FLAG = true;
        }
        auto blob = createSharedRef<ImageBlob>();
        m_preprocessor->Run(img, blob);
        InferResults(blob, result);
//        if (m_config->TIMING) {
//            auto timing = Util::toc();
//            if (timing > 0) {
//                m_curr_fps = 1000.0f / (float) timing;
//                std::cout << "Current FPS: " << m_curr_fps << std::endl;
//            }
//        }

    }

    void TrtDeploy::Init(const std::string &model_file) {
        std::ifstream ai_model(model_file, std::ios::in | std::ios::binary);
        if (!ai_model) {
            std::cerr << "Read serialized file: " << model_file << " failed" << std::endl;
            m_model_load_status = ModelLoadStatus::LOADED_FAILED;
            return;
        }
        auto mSize = Util::getFileSize(model_file);
        std::vector<char> buf(mSize);
        ai_model.read(&buf[0], mSize);
        ai_model.close();
        std::cout << "Model size: " << mSize << std::endl;
        m_config->MODEL_NAME = model_file;
        m_model_load_status = ModelLoadStatus::LOADED_SUCCESS;
        if (!m_logger) {
            m_logger = createSharedRef<Logger>();
        }
        if (!m_preprocessor) {
            m_preprocessor = createSharedRef<Preprocessor>(m_config);
        }
        if (!m_postprocessor) {
            m_postprocessor = createSharedRef<Postprocessor>(m_config);
        }
        if (!m_runtime) {
            m_runtime = nvinfer1::createInferRuntime(*m_logger);
            initLibNvInferPlugins(m_logger.get(), "");
            m_engine = m_runtime->deserializeCudaEngine((void *) &buf[0], mSize);
            m_execution_context = m_engine->createExecutionContext();
        }

        auto in_dims = m_engine->getTensorShape(m_config->INPUT_NAME.c_str());
        ///note the target size should match the model input.
        assert(m_config->TARGET_SIZE[1] == in_dims.d[in_dims.nbDims - 1] &&
               m_config->TARGET_SIZE[0] == in_dims.d[in_dims.nbDims - 2]);
        int in_size = 1;
        for (int i = 0; i < in_dims.nbDims; ++i) {
            in_size *= in_dims.d[i];
        }
        m_output_state.resize(m_config->OUTPUT_NAMES.size());
        m_device_ptr.resize(1 + m_config->OUTPUT_NAMES.size());//with input pointer, thus+1.
        for (int i = 0; i < m_config->OUTPUT_NAMES.size(); ++i) {
            auto out_dims_i = m_engine->getTensorShape(m_config->OUTPUT_NAMES[i].c_str());
            int out_size = 1;
            for (int j = 0; j < out_dims_i.nbDims; ++j) {
                out_size *= out_dims_i.d[j];
            }
            m_output_state[i].resize(out_size);
        }

        cudaError_t state;
        state = cudaMalloc(&m_device_ptr[0], in_size * sizeof(float));
        if (state) {
            std::cout << "Allocate memory failed" << std::endl;
            m_cuda_alloc_status = CudaMemAllocStatus::ALLOC_FAILED;
            return;
        }
        for (int i = 0; i < m_config->OUTPUT_NAMES.size(); ++i) {
            state = cudaMalloc(&m_device_ptr[i + 1],
                               m_output_state[i].size() * sizeof(float));
            if (state) {
                std::cout << "Allocate memory failed" << std::endl;
                m_cuda_alloc_status = CudaMemAllocStatus::ALLOC_FAILED;
                return;
            }
        }
        m_cuda_alloc_status = CudaMemAllocStatus::ALLOC_SUCCESS;
        state = cudaStreamCreate(&m_stream);
        if (state) {
            std::cout << "Create stream failed" << std::endl;
            return;
        }
        m_execution_context->setTensorAddress(m_config->INPUT_NAME.c_str(),
                                              m_device_ptr[0]);
        for (int i = 0; i < m_config->OUTPUT_NAMES.size(); ++i) {
            m_execution_context->setTensorAddress(m_config->OUTPUT_NAMES[i].c_str(),
                                                  m_device_ptr[i + 1]);
        }

        int w = m_config->TARGET_SIZE[m_config->TARGET_SIZE.size() - 1];
        int h = m_config->TARGET_SIZE[m_config->TARGET_SIZE.size() - 2];
        auto *ptr = (float *) m_device_ptr[0];
        for (int i = 0; i < m_config->TRIGGER_LEN; ++i) {
            for (int j = 0; j < 3; ++j) {
                m_cv_data.emplace_back(cv::Size(w, h), CV_32FC1, ptr + j * w * h + i * 3 * w * h);
            }
        }
    }

    TrtDeploy::ModelLoadStatus TrtDeploy::LoadStatus() {
        return m_model_load_status;
    }

    TrtDeploy::CudaMemAllocStatus TrtDeploy::MemAllocStatus() {
        return m_cuda_alloc_status;
    }

    void TrtDeploy::Warmup(SharedRef<TrtResults> &res) {
        cv::Mat img = cv::Mat::ones(cv::Size(m_config->INPUT_SHAPE[m_config->INPUT_SHAPE.size() - 1],
                                             m_config->INPUT_SHAPE[m_config->INPUT_SHAPE.size() - 2]), CV_8UC3);
        std::vector<cv::Mat> temp(m_config->TRIGGER_LEN);
        for (int i = 0; i < m_config->TRIGGER_LEN; ++i) {
            temp[i] = img.clone();
        }
        for (int i = 0; i < 10; i++)
            Infer(temp, res);
    }

    void TrtDeploy::InferResults(SharedRef<ImageBlob> &data, SharedRef<TrtResults> &res) {
        res->Clear();

        for (int i = 0; i < data->m_gpu_data.size(); ++i) {
            cv::cuda::split(data->m_gpu_data[i], &m_cv_data[3 * i]);
        }

        m_execution_context->enqueueV3(m_stream);
        for (int i = 0; i < m_config->OUTPUT_NAMES.size(); ++i) {
            auto state = cudaMemcpyAsync(&m_output_state[i][0], m_device_ptr[i + 1],
                                         m_output_state[i].size() * sizeof(float),
                                         cudaMemcpyDeviceToHost, m_stream);
            if (state) {
                std::cout << "Transmit to host failed." << std::endl;
            }
        }
        //warp the state to our res.
        int idx = 0;
        for (auto &name: m_config->OUTPUT_NAMES) {
            res->Set(std::make_pair(name, m_output_state[idx++]));
        }
    }

    void TrtDeploy::Postprocessing(const SharedRef<TrtResults> &res, const std::vector<cv::Mat> &img,
                                   std::vector<cv::Mat> &out_img,int& alarm) {
        m_postprocessor->Run(res, img, out_img,alarm);
    }

}
