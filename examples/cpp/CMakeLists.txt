cmake_minimum_required(VERSION 3.27)

set(TRTLLM_DIR "/code/tensorrt_llm")
list(APPEND CMAKE_MODULE_PATH "${TRTLLM_DIR}/cpp/cmake/modules")

if(NOT TRTLLM_BUILD_DIR)
  set(TRTLLM_BUILD_DIR "${TRTLLM_DIR}/cpp/build")
endif()
set(TRTLLM_LIB_PATH "${TRTLLM_BUILD_DIR}/tensorrt_llm/libtensorrt_llm.so")
set(TRTLLM_PLUGIN_PATH
    "${TRTLLM_BUILD_DIR}/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so"
)
set(TRTLLM_INCLUDE_DIR "${TRTLLM_DIR}/cpp/include")
set(TRTLLM_INCLUDE_DIR_PLUGIN "${TRTLLM_DIR}/cpp/")

# Determine CXX11 ABI compatibility
execute_process(
  COMMAND bash -c "nm -f posix -D ${TRTLLM_LIB_PATH} | grep __cxx11"
  RESULT_VARIABLE GLIB_CXX11_FOUND
  OUTPUT_QUIET)
if(GLIB_CXX11_FOUND EQUAL 0)
  set(USE_CXX11_ABI 1)
else()
  set(USE_CXX11_ABI 0)
endif()
message(STATUS "Use CXX11 ABI: ${USE_CXX11_ABI}")
add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_VERBOSE_MAKEFILE 1)

# Define project name
project(executorExamples)

# Compile options
set(CMAKE_CXX_FLAGS "-Wall -pthread -lstdc++ -DENABLE_MULTI_DEVICE=1")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE release)

find_package(CUDAToolkit REQUIRED COMPONENTS cuda_driver cudart_static nvml)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")

# TRT dependencies
find_package(TensorRT 10 REQUIRED)


if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message(
    STATUS
      "CUDA_VERSION ${CUDA_VERSION} is greater or equal than 11.0, enable -DENABLE_BF16 flag"
  )
endif()

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(
    STATUS
      "CUDA_VERSION ${CUDA_VERSION} is greater or equal than 11.8, enable -DENABLE_FP8 flag"
  )
endif()


set(TORCH_DIR "/usr/local/lib/python3.12/dist-packages/torch")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)
add_compile_definitions(TORCH_CUDA=1)

include_directories(${TRTLLM_INCLUDE_DIR} ${TRTLLM_INCLUDE_DIR_PLUGIN} ${CUDAToolkit_INCLUDE_DIRS} /usr/local/tensorrt/include ${TORCH_INCLUDE_DIRS})



link_directories(/code/tensorrt_llm/build/lib.linux-x86_64-cpython-312/tensorrt_llm/libs/)
add_executable(demo_multi_stream main_multi-stream.cpp)
target_link_libraries(demo_multi_stream tensorrt_llm nvinfer_plugin_tensorrt_llm ${TORCH_LIBRARIES}  ${Python3_LIBRARIES})