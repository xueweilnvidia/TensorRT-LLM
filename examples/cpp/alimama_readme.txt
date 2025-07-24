cd examples/models/core/qwen

python convert_checkpoint.py --model_dir /llm-models/Qwen2.5-0.5B-Instruct \
                              --output_dir ./tllm_checkpoint_1gpu_fp16 \
                              --dtype float16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16 \
            --output_dir ./tmp/qwen/Qwen2.5-0.5B-Instruct/trt_engines/fp16/1-gpu \
            --gemm_plugin float16


cd /code/tensorrt_llm/examples/cpp

mkdir build
cd build
cmake ..
make -j

./demo_multi_stream   "forward" /code/tensorrt_llm/examples/models/core/qwen/tmp/qwen/Qwen2.5-0.5B-Instruct/trt_engines/fp16/1-gpu/rank0.engine  2 1 100 512