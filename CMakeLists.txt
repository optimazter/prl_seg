cmake_minimum_required(VERSION 3.16)

project(prl_seg VERSION 0.1 LANGUAGES CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MATLAB_DIR /usr/local/MATLAB/R2024b)

set(SimpleITK_DIR "/home/adrian-hjertholm-voldseth/dev/SimpleITK-build/SimpleITK-build")
find_package(SimpleITK REQUIRED)

set(FSL_DIR "/usr/local/fsl")

set(JULIA_DIR "/home/adrian-hjertholm-voldseth/dev/julia-1.11.1")
set(JULIA_LIBRARY ${JULIA_DIR}/lib/libjulia.so)

set(Torch_DIR "/home/adrian-hjertholm-voldseth/dev/libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(PROJECT_SOURCES 
    preprocess.cpp
    lib/dataset.h
    lib/dataset.cpp
    lib/QSM.h
    lib/QSM.cpp
    lib/romeo.h
    lib/romeo.cpp
)

configure_file(env.h.in ${CMAKE_CURRENT_SOURCE_DIR}/lib/env.h @ONLY) 

add_executable(prl_seg ${PROJECT_SOURCES})

target_link_libraries(prl_seg 
    ${MATLAB_DIR}/extern/bin/glnxa64/libMatlabDataArray.so
    ${MATLAB_DIR}/extern/bin/glnxa64/libMatlabEngine.so
    ${SimpleITK_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${JULIA_LIBRARY}
)

target_include_directories(prl_seg PUBLIC
    ${MATLAB_DIR}/extern/include
    ${JULIA_DIR}/include/julia
    qsm_pipeline
    lib
)


file(COPY qsm_pipeline DESTINATION ${CMAKE_BINARY_DIR})
