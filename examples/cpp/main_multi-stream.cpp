#include <cstdint>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <iostream>
#include <fstream>
#include <map>

#include <NvInfer.h>
#include <NvInferRuntimeBase.h>
#include <c10/core/TensorOptions.h>
#include <torch/types.h>
#include <torch/script.h>

#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/cudaStream.h"
// #include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/torchView.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

namespace trt = nvinfer1;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm;

// // 保存张量到文件（保留原功能）
// void savetensor(void* data, int size, std::string name, torch::TensorOptions opts) {
//     torch::Tensor tensor = torch::from_blob(data, {size}, opts);
//     auto pickled = torch::jit::pickle_save(tensor);
//     std::ofstream fout(name, std::ios::out | std::ios::binary);
//     fout.write(pickled.data(), pickled.size());
//     fout.close();
// }

// 性能统计结果结构体
struct BenchmarkResult {
    double avg_ttft;     // 平均首token时间 (秒)
    double throughput;   // QPS
    int input_len;
    int batch_size;
    int num_threads;
};

// 线程安全的性能数据收集器
class PerformanceCollector {
public:
    void addLatency(double latency_sec) {
        std::lock_guard<std::mutex> lock(mtx_);
        latencies_.push_back(latency_sec);
    }

    std::vector<double> getLatencies() {
        std::lock_guard<std::mutex> lock(mtx_);
        return latencies_;
    }

private:
    std::vector<double> latencies_;
    std::mutex mtx_;
};

class OutputCollector {
public:
    struct ThreadOutput {
        std::vector<float> logits;
        std::vector<float> hidden_states;
    };

    void addOutput(int thread_id, int iteration, 
                  const std::vector<float>& logits_data,
                  const std::vector<float>& hidden_states_data) {
        std::lock_guard<std::mutex> lock(mtx_);
        outputs_[thread_id][iteration] = {logits_data, hidden_states_data};
    }

    const std::map<int, std::map<int, ThreadOutput>>& getOutputs() const {
        return outputs_;
    }

private:
    std::map<int, std::map<int, ThreadOutput>> outputs_; // thread_id -> iteration -> output
    std::mutex mtx_;
};

int64_t getTensorSize(const nvinfer1::Dims& shape) {
    int64_t size = 1;
    for (int i = 0; i < shape.nbDims; ++i) {
        size *= shape.d[i];
    }
    return size;
}


// 推理工作线程
void inferenceWorker(
    TllmRuntime& rt, 
    nvinfer1::ICudaEngine& engine,
    int thread_id,
    int batch_size,
    int input_len,
    PerformanceCollector& collector,
    OutputCollector& output_collector,
    const torch::Tensor& fixed_input_ids,
    const torch::Tensor &fixed_position_ids,
    const torch::Tensor &fixed_last_token_ids,
    int num_iterations = 100,
    int num_warmup = 10) {
    
    auto stream = std::make_shared<CudaStream>();
    
    auto device_id = stream->getDevice();
    auto const nbIoTensors = engine.getNbIOTensors();
    auto& allocator = rt.getBufferManager();

    TllmRuntime::TensorMap inputTensorMap{};
    TllmRuntime::TensorMap outputTensorMap{};

    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_id);
    auto f16_opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, device_id);
    
    int seq_len = input_len; // 使用传入的输入长度
    // apply i/o tensor for fake run.
    for (int i = 0; i < nbIoTensors; i++)
    {
        auto const name = engine.getIOTensorName(i);
        auto dims = engine.getTensorShape(name);
        dims.d[0] = batch_size;
        if(strcmp("hidden_states_output", name) == 0 || 
                strcmp("input_ids", name) == 0 || 
                strcmp("position_ids", name) == 0) {
            dims.d[0] = batch_size * seq_len; 
        } else if(strcmp("cache_indirection", name) == 0) {
            dims.d[1] = 1; 
            dims.d[2] = 32768; 
        } else if(strstr(name, "host_max_attention_window_size") != NULL) {
            dims.d[0] = 24; 
        }else if(strstr(name, "host_sink_token_length") != NULL){
            dims.d[0] = 1;
        }else if(strcmp(name, "host_kv_cache_pool_mapping") == 0) {
            dims.d[0] = 24;  // num_layers
            dims.d[1] = 2;  // 2 
        }
        else if(strcmp(name, "host_kv_cache_pool_pointers") == 0){
            dims.d[0] = 1;  // num_layers
            dims.d[1] = 2;  // 2 
        }
        if(strcmp(name, "kv_cache_block_offsets") == 0 || 
            strcmp(name, "host_kv_cache_block_offsets") == 0) {
            dims.d[0] = 1;  // num_layers
            dims.d[1] = 1;  // max_batch_size
            dims.d[2] = 2;  // 2 
            dims.d[3] = 1024;  //max_blocks_per_sequence * 2
            // std::cout << "kv_cache_block_offsets dims : " << "24 1 2 seqlen+1" << std::endl;
        }
        if(strcmp(name, "host_max_attention_window_sizes") == 0) {
            dims.d[0] = 24;  // 2 
        }
        
        
        std::cout << "name: " << name << ", dims: " << ITensor::toString(dims) << std::endl;

        auto const type = rt.getEngine().getTensorDataType(name);
        if(strcmp("logits", name) == 0 || strcmp("hidden_states_output", name) == 0) {
            continue;
        } 
        // 创建输入张量
        ITensor::SharedPtr buffer = nullptr;
        at::Tensor t;
        std::vector<int64_t> shape;
        for (int j = 0; j < dims.nbDims; ++j)
            shape.push_back(dims.d[j]);

        if(strcmp("position_ids", name) == 0) {
            // t = torch::randint(1, 2, shape, i32_opts); 
            t = fixed_position_ids.clone(); // 克隆一份，避免多线程共享问题
        }
        else if(strcmp("sequence_length", name) == 0) {
            std::vector<int> data(batch_size);
            for (int j = 0; j < batch_size; j++) {
                data[j] = seq_len ;
            }
            t = torch::tensor(data, i32_opts);
        }
        else if(strcmp("input_ids", name) == 0) {
            std::cout << "name: " << name << ", dims: " << ITensor::toString(dims) << std::endl;            
            // t = torch::randint(1000, 5000, {batch_size*seq_len}, i32_opts); 
            t = fixed_input_ids.clone();  
        }
        else if(strcmp("last_token_ids", name) == 0) {
            // t = torch::randint(1000, 1001, {batch_size}, i32_opts); 
            t = fixed_last_token_ids.clone();
        }
        else if(strcmp("host_max_attention_window_sizes", name) == 0) {
            std::vector<int> data(24);
            for (int i = 0; i < 24; i++){
                data[i] = 24;
            }
            t = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32));
        }
        else if(strstr(name, "host_runtime_perf_knobs") != NULL) {
            int perf_knob_tensor_size = 16;
            std::vector<int> data(perf_knob_tensor_size, -1);
            t = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt64));
        }
        else if(strstr(name, "host_context_progress") != NULL) {
            std::vector<int> data{0};
            t = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt64));
        }
        else if(strstr(name, "kv_cache_block_offsets") != NULL) {
            t = torch::randint(1, 2, shape, i32_opts); 
        }
        else if(strstr(name, "host_kv_cache_pool_pointers") != NULL) {
             std::vector<int> data(2);
            for (int j = 0; j < batch_size; j++) {
                data[j] = 0; 
            }
            t = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt64));
            t = t.reshape({1, 2}); // 24 layers, 2 pointers
        }
        else if(strstr(name, "host_kv_cache_pool_mapping") != NULL) {
             std::vector<int> data(48);
            for (int j = 0; j < batch_size; j++) {
                data[j] = 0; 
            }
            t = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32));
            t = t.reshape({24, 2}); // 24 layers, 2 pointers
        }
        else if(strstr(name, "host_kv_cache_block_offsets") != NULL) {
            t = torch::randint(0, 1, shape, i32_opts); 
        }
        else if(strcmp("context_lengths", name) == 0) {
            std::vector<int> data(batch_size);
            for (int j = 0; j < batch_size; j++) {
                data[j] = seq_len; 
            }
            t = torch::tensor(data, i32_opts);
        } 
        else if(strcmp("host_context_lengths", name) == 0) {
            std::vector<int> data(batch_size);
            for (int j = 0; j < batch_size; j++) {
                data[j] = seq_len; 
            }
            t = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32));
        } else if(strcmp("host_past_key_value_lengths", name) == 0) {
            std::vector<int> data(batch_size);
            for (int j = 0; j < batch_size; j++) {
                data[j] = 0; 
            }
            t = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32));
        } else if(strcmp("host_sink_token_length", name) == 0) {
            std::vector<int> data{0};
            t = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32));
        } else if(strcmp("host_request_types", name) == 0) { 
            std::vector<int> data(batch_size);
            for (int j = 0; j < batch_size; j++) {
                data[j] = 0; // prefill=0 deocde=1
            }
            t = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32));
        } else if(strcmp("cache_indirection", name) == 0) {
            // int* d_ptr = d_i32_ptr_ + (max_batch_size_ * max_seq_len_) + max_batch_size_ * 3;
            // t = torch::from_blob(d_ptr, shape, i32_opts);
            t = torch::randint(1, 2, shape, i32_opts); 
        } else if(strstr(name, "past_key_value_") != NULL) {
            t = torch::randn(shape, f16_opts); 
            // t = torch::from_blob(kv_cache_[kv_idx], shape, input_opts);
            // kv_idx += 1;
        } else if(strstr(name, "present_key_value_") != NULL) {
            t = torch::randn(shape, f16_opts); 
            // t = torch::from_blob(kv_cache_[kv_idx], shape, input_opts);
            // kv_idx += 1;
        } 
        else {
            // 添加详细日志
            std::string type_str;
            switch (type) {
                case trt::DataType::kFLOAT:  type_str = "float32"; break;
                case trt::DataType::kHALF:   type_str = "float16"; break;
                case trt::DataType::kINT8:   type_str = "int8";    break;
                case trt::DataType::kINT32:  type_str = "int32";   break;
                case trt::DataType::kBF16:   type_str = "bfloat16"; break;
                default:                     type_str = "unknown";  continue;
            }
            std::cout<<"name: "<<name<<std::endl;
            std::cout<< "type_str: "<< type_str.c_str() << std::endl;
            throw std::runtime_error("dtype: only support 'half' 'bfloat16' and 'int32'");
        }

        // auto buffer = std::shared_ptr<ITensor>{allocator.gpu(dims, type)};
        // allocator.setZero(*buffer);


        // allocate device memory from torch tensor
       if (strcmp("input_ids", name) == 0 || strcmp("position_ids", name) == 0 || strcmp("last_token_ids", name) == 0
    || strcmp("kv_cache_block_offsets", name) == 0 || strcmp("sequence_length", name) == 0
    || strcmp("context_lengths", name) == 0 || strcmp("cache_indirection", name) == 0 || strcmp("logits", name) == 0){
        t = t.to(torch::kCUDA).contiguous();
    }else{
        t = t.to(torch::kCPU).contiguous();
    }

        
        // t = t.to(torch::kCUDA).contiguous();
        buffer = tensorrt_llm::runtime::TorchView::of(t);
        if (rt.getEngine().getTensorIOMode(name) == trt::TensorIOMode::kINPUT) {
            inputTensorMap.insert(std::make_pair(name, buffer));
        }
        else if (rt.getEngine().getTensorIOMode(name) == trt::TensorIOMode::kOUTPUT) {
            outputTensorMap.insert(std::make_pair(name, buffer));
        }
    }
    rt.setInputTensors(thread_id, inputTensorMap);
    rt.getStream().synchronize();
    rt.setOutputTensors(thread_id, outputTensorMap);
    rt.getStream().synchronize();
    TLLM_LOG_INFO("Set input/output tensor success");
    TLLM_LOG_INFO("Warm-up  begin ");
    cudaDeviceSynchronize(); 
    // **Warm-up 预热阶段**
    // num_warmup = 10;
    for (int warmup = 0; warmup < num_warmup; ++warmup) {
        rt.executeContextWithStream(thread_id, stream->get());
	    stream->synchronize();
        // rt.executeContext(thread_id);
        // rt.getStream().synchronize();
        // cudaDeviceSynchronize(); 
    }
    // cudaDeviceSynchronize();
    TLLM_LOG_INFO("Warm-up  success");

    for (int iter = 0; iter < num_iterations; ++iter) {
        // TLLM_LOG_INFO("Infer , ite: %d", iter);
        auto start = std::chrono::high_resolution_clock::now();
        auto ret = rt.executeContextWithStream(thread_id, stream->get());
        stream->synchronize();
        // rt.executeContext(thread_id);
        // rt.getStream().synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        // 收集logits输出
        std::vector<float> host_logits;
        auto logits_tensor = outputTensorMap["logits"].get();
        int size = 1;
        auto shape = logits_tensor->getShape();
        size = getTensorSize(shape);
        host_logits.resize(size);
        cudaMemcpy(host_logits.data(), logits_tensor->data(), size * sizeof(float), cudaMemcpyDeviceToHost);
        // 打印前 100 个元素
        //int print_count = std::min(10, static_cast<int>(host_logits.size()));
        //std::cout << "First " << print_count << " elements of host_logits:" << std::endl;
        //for (int i = 0; i < print_count; ++i) {
        //    std::cout << host_logits[i] << " ";
        //    if ((i + 1) % 10 == 0) std::cout << std::endl; // 每 10 个元素换行，提高可读性
        //}
        // std::cout << std::endl;    

        // 收集hidden_states输出
        std::vector<float> host_hidden;
        //auto hidden_states_output_tensor = outputTensorMap["hidden_states_output"].get();
        //shape = hidden_states_output_tensor->getShape();
        //size = getTensorSize(shape);
        //host_hidden.resize(size); 
        //cudaMemcpy(host_hidden.data(), hidden_states_output_tensor->data(), size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 保存结果
        output_collector.addOutput(thread_id, iter, host_logits, host_hidden);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
        collector.addLatency(duration);

    }
    TLLM_LOG_INFO("infer success!");
}

void analyzeConsistency(const OutputCollector& collector, int input_len,const std::string& log_path) {
    auto results = collector.getOutputs();
    std::ofstream log(log_path, std::ios_base::app);
    
    log << "[Result Consistency Analysis]\n";
    if (results.empty()) {
        log << "No results collected.\n";
        return;
    }

    // 以第一个结果为基准
    auto& base = results.begin()->second.begin()->second;
    log<< " -----input_len= "<< input_len<< "----------- \n";
    for (const auto& thread : results) {
        for (const auto& iter : thread.second) {
            // 检查logits一致性
            bool logits_ok = (iter.second.logits.size() == base.logits.size());
            if (logits_ok) {
                for (size_t i = 0; i < base.logits.size(); ++i) {
                    if (std::abs(base.logits[i] - iter.second.logits[i]) > 1e-5f) {
                        std::cout<< base.logits[i] << " vs " << iter.second.logits[i] << std::endl;
                        logits_ok = false;
                        break;
                    }
                }
            }

            // 检查hidden states一致性
            bool hidden_ok = (iter.second.hidden_states.size() == base.hidden_states.size());
            if (hidden_ok) {
                for (size_t i = 0; i < base.hidden_states.size(); ++i) {
                    if (std::abs(base.hidden_states[i] - iter.second.hidden_states[i]) > 1e-5f) {
                        hidden_ok = false;
                        break;
                    }
                }
            }
            if (thread.first !=0){
                log << "Thread " << thread.first << " Iter " << iter.first << ": "
                    << "Logits=" << (logits_ok ? "OK" : "DIFF") << ", "
                    << "Hidden=" << (hidden_ok ? "OK" : "DIFF") << "\n";
            }
        }
    }
    log << "-------------------------\n";
}
// 执行基准测试
BenchmarkResult runBenchmark(
    const std::string& model_path,
    const std::string& function_name,
    int batch_size,
    int input_len,
    int num_threads,
    int num_iterations,
    int num_warmup = 10) {
    OutputCollector output_collector;
    std::cout<< "rt init begin"<< std::endl;
    auto logger = std::make_shared<TllmLogger>();
    TllmRuntime rt {RawEngine(model_path), logger.get(),};
    // rt.loadManagedWeights(RawEngine(model_path), 0);
    std::cout<< "rt init"<< std::endl;
    nvinfer1::ICudaEngine& engine = rt.getEngine();

    // 为每个线程添加上下文
    for (int i = 0; i < num_threads; ++i) {
        rt.addContext(0);
    }
    rt.getStream().synchronize();

    PerformanceCollector collector;
    std::vector<std::thread> threads;
    // 在 runBenchmark 函数中，TllmRuntime 初始化之后添加：
    int total_input_size = batch_size * input_len;
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor fixed_input_ids = torch::randint(1000, 5000, {total_input_size}, i32_opts); 
    torch::Tensor fixed_position_ids = torch::arange(0, input_len, i32_opts).repeat({batch_size});
    torch::Tensor fixed_last_token_ids = torch::randint(1000, 5000, {batch_size}, i32_opts);
    auto start_time = std::chrono::high_resolution_clock::now();

    // 启动工作线程
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(inferenceWorker, 
                             std::ref(rt), 
                             std::ref(engine), 
                             i, 
                             batch_size, 
                             input_len, 
                             std::ref(collector), 
                             std::ref(output_collector),
                             std::cref(fixed_input_ids),
                             std::cref(fixed_position_ids),
                             std::cref(fixed_last_token_ids),
                             num_iterations, 
                             num_warmup);
    }
    // inferenceWorker(std::ref(rt), 
    //                          std::ref(engine), 
    //                          0, 
    //                          batch_size, 
    //                          input_len, 
    //                          std::ref(collector), 
    //                          std::ref(output_collector),
    //                          std::cref(fixed_input_ids),
    //                          std::cref(fixed_position_ids),
    //                          std::cref(fixed_last_token_ids),
    //                          num_iterations, 
    //                          num_warmup);

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    // 分析结果一致性
    analyzeConsistency(output_collector, input_len, "consistency_log.txt");

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()/1e3;
    std::cout<< "Total duration: " << total_duration << " seconds" << std::endl;

    // 计算性能指标
    auto latencies = collector.getLatencies();
    double avg_ttft = 0.0;
    double duration_ = 0.0;
    if (!latencies.empty()) {
        for (auto l : latencies){
            avg_ttft += l;
            duration_ +=l;
        }
        avg_ttft /= latencies.size();
    }

    int total_requests = num_threads * num_iterations;
    double qps = (duration_ > 0) ? (total_requests / duration_)*1000*num_threads: 0.0;

    return {avg_ttft, qps, input_len, batch_size, num_threads};
}

int main(int argc, char** argv) {
    // 参数解析（示例）
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <function_name> <model_path> <num_threads> <batch_size> <loop> [input_lengths...]"
                  << std::endl;
        return 1;
    }

    const char* func_name = argv[1];
    const char* model_path = argv[2];
    int num_threads = std::stoi(argv[3]);
    int batch_size = std::stoi(argv[4]);
    int num_iterations = std::stoi(argv[5]);
    // 支持多个输入长度测试
    std::vector<int> input_lengths;
    for (int i = 6; i < argc; ++i) {
        input_lengths.push_back(std::stoi(argv[i]));
    }

    if (input_lengths.empty()) {
        input_lengths = {1, 128, 256, 512, 1024, 2048, 4096}; // 默认测试长度
    }
    // input_lengths = {1};

    // 初始化TensorRT-LLM插件
    std::shared_ptr<nvinfer1::ILogger> mTrtLogger = std::make_shared<tensorrt_llm::runtime::TllmLogger>();
    initTrtLlmPlugins(mTrtLogger.get());
    std::cout << "Load plugins success" << std::endl;
    // **CSV 文件初始化**
    std::ofstream csv_file("benchmark_results.csv");
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file for writing." << std::endl;
        return 1;
    }
    // 写入 CSV 表头
    csv_file << "Input_Length,Batch_Size,Num_Threads,Avg_TTFT,QPS\n";
    std::cout<< "write csv file head!" << std::endl;
    // 每个输入长度执行基准测试
    for (int len : input_lengths) {
        auto result = runBenchmark(model_path, func_name, batch_size, len, num_threads, num_iterations);

        std::cout << "[Benchmark Result]" << std::endl;
        std::cout << "num_iterations: " << num_iterations << std::endl;
        std::cout << "Input Length: " << result.input_len << std::endl;
        std::cout << "Batch Size: " << result.batch_size << std::endl;
        std::cout << "Num Threads: " << result.num_threads << std::endl;
        std::cout << "Avg TTFT (ms): " << result.avg_ttft << std::endl;
        std::cout << "Throughput (QPS): " << result.throughput << std::endl;
        std::cout << "-----------------------------" << std::endl;

        // 写入 CSV 文件
        csv_file << len << "," 
                 << batch_size << "," 
                 << num_threads << "," 
                 << std::fixed << std::setprecision(4) << result.avg_ttft << ","
                 << std::fixed << std::setprecision(2) << result.throughput << "\n";
    }
    csv_file.close();
    std::cout << "Benchmark results written to 'benchmark_results.csv'" << std::endl;
    return 0;
}
