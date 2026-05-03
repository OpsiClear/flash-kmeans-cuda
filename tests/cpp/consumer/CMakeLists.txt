cmake_minimum_required(VERSION 3.24)

project(flash_kmeans_cuda_consumer_smoke LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(flash_kmeans_cuda CONFIG REQUIRED)

add_executable(consumer_smoke main.cpp)
target_link_libraries(consumer_smoke PRIVATE flash_kmeans_cuda::flash_kmeans_cuda)
target_compile_features(consumer_smoke PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(consumer_smoke PRIVATE /O2 /EHsc /bigobj /Zc:preprocessor)
else()
  target_compile_options(consumer_smoke PRIVATE -O3)
endif()
