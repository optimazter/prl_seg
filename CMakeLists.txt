cmake_minimum_required(VERSION 3.16)

project(prl_seg VERSION 0.1 LANGUAGES CXX C)


set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Torch_DIR "/home/adrian-hjertholm-voldseth/dev/libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(ITK_DIR "/home/adrian-hjertholm-voldseth/dev/SimpleITK-build/ITK-build")
set(SimpleITK_DIR "/home/adrian-hjertholm-voldseth/dev/SimpleITK-build/SimpleITK-build")
find_package(SimpleITK REQUIRED)

find_package(Boost COMPONENTS system filesystem REQUIRED)

set(JULIA_DIR "/home/adrian-hjertholm-voldseth/dev/julia-1.11.1")
set(JULIA_LIBRARY ${JULIA_DIR}/lib/libjulia.so)

set(PROJECT_SOURCES
        main.cpp
        lib/cmd.h
        lib/cmd.cpp
        lib/fsl.h
        lib/fsl.cpp
        lib/medi.h
        lib/medi.cpp
        lib/c_romeo.h
        lib/romeo.h
        lib/romeo.cpp
        lib/sitk_utils.h
        lib/sitk_utils.cpp
)


set (PROJECT_LIBS
        libs/FSL/fsl_cmd.sh
)


add_executable(prl_seg 
    ${PROJECT_SOURCES}
    ${PROJECT_LIBS}
)

target_include_directories(prl_seg PRIVATE
    lib
    libs
    ${JULIA_DIR}/include/julia
)

file(COPY libs DESTINATION ${CMAKE_BINARY_DIR})

target_link_libraries(prl_seg PRIVATE
    ${TORCH_LIBRARIES}
    ${SimpleITK_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${JULIA_LIBRARY}
    
)