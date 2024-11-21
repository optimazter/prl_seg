cmake_minimum_required(VERSION 3.16)

project(prl_seg_master VERSION 0.1 LANGUAGES CXX C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)



add_subdirectory(prl_seg)
add_executable(prl_seg_master main.cpp)


target_link_libraries(prl_seg_master PRIVATE PRLSeg)