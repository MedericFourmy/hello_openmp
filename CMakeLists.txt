cmake_minimum_required(VERSION 3.10)

project(hello_openmp)

find_package(OpenMP)

add_executable(main src/main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
endif()