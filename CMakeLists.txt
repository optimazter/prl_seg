cmake_minimum_required(VERSION 3.16)

project(prl_seg VERSION 0.1 LANGUAGES CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_QSM "Build QSM" OFF)


set(ITK_DIR "C:/Users/adria/dev/libs/SimpleITK-build/ITK-build")
find_package(ITK REQUIRED)

set(SimpleITK_DIR "C:/Users/adria/dev/libs/SimpleITK-build/SimpleITK-build")
find_package(SimpleITK REQUIRED)


set(Torch_DIR "C:/Users/adria/dev/libs/libtorch-win-shared-with-deps-debug-2.5.1+cpu/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


set(PROJECT_SOURCES 
    preprocess.cpp
    lib/dataset.h
    lib/dataset.cpp
)

if (BUILD_QSM)
    list(APPEND PROJECT_SOURCES  
        lib/QSM.h
        lib/QSM.cpp
        lib/romeo.h
        lib/romeo.cpp
    )
endif(BUILD_QSM)

configure_file(env.h.in ${CMAKE_CURRENT_SOURCE_DIR}/lib/env.h @ONLY) 

add_executable(prl_seg ${PROJECT_SOURCES})

target_link_libraries(prl_seg 
    ${SimpleITK_LIBRARIES} 
    ${TORCH_LIBRARIES}
)

target_include_directories(prl_seg PUBLIC
    lib
)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET prl_seg
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:prl_seg>)
endif (MSVC)

if (BUILD_QSM)
    add_subdirectory(qsm_pipeline)
    set(MATLAB_DIR /home/adrian/dev/libs/matlab)
    set(FSL_DIR "/home/adrian/dev/libs/fsl")
    set(JULIA_DIR "/home/adrian/dev/libs/julia-1.11.2")
    set(JULIA_LIBRARY ${JULIA_DIR}/lib/libjulia.so)

    target_link_libraries(prl_seg
        ${MATLAB_DIR}/extern/bin/glnxa64/libMatlabDataArray.so
        ${MATLAB_DIR}/extern/bin/glnxa64/libMatlabEngine.so
        ${JULIA_LIBRARY}
    )

    target_include_directories(prl_seg PUBLIC
        ${MATLAB_DIR}/extern/include
        ${JULIA_DIR}/include/julia
        qsm_pipeline
    )

    file(COPY qsm_pipeline DESTINATION ${CMAKE_BINARY_DIR})

endif(BUILD_QSM)
