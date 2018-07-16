cmake_minimum_required(VERSION 3.5)
project(darknet-fork)

find_package(PkgConfig)

set(CMAKE_C_STANDARD 99)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)

include_directories(src include)
FILE(GLOB SOURCE_FILES src/*.c src/*.h )
FILE(GLOB EXAMPLES examples/*.c)
FILE(GLOB INCLUDE include/*.h)

set(SOURCE_FILES ${SOURCE_FILES} ${EXAMPLES} ${INCLUDE})

set(COMMON_FLAGS "-Wall -Wno-unknown-pragmas -Wfatal-errors")

set(LINKER_LIBS pthread m)

if (CUDNN)
    add_definitions(-DCUDNN)
    set(LINKER_LIBS ${LINKER_LIBS} cudnn)
endif ()

if (OPENCV)
    find_package(OpenCV REQUIRED)
    add_definitions(-DOPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
    #    set(COMMON_FLAGS "${COMMON_FLAGS} ${OpenCV_INCLUDE_DIRS}")
    pkg_search_module(OPENCV3 REQUIRED opencv)

    link_directories(${OPENCV3_LIBRARY_DIRS})
    set(LINKER_LIBS ${LINKER_LIBS} ${OPENCV3_LIBRARIES})

    #    message(${OPENCV3_LIBRARIES})
    # loop through all env variables
    #    get_cmake_property(_variableNames VARIABLES)
    #    foreach (_variableName ${_variableNames})
    #        message(STATUS "${_variableName}=${${_variableName}}")
    #    endforeach()
endif ()

if (OPENMP)
    set(COMMON_FLAGS "${COMMON_FLAGS} -fopenmp")
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND (OPENCV))
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Og")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${COMMON_FLAGS}")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${COMMON_FLAGS} -Ofast")
else ()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS} -Ofast")
endif ()

if (GPU)
    #https://stackoverflow.com/questions/13073717/building-cuda-object-files-using-cmake
    FILE(GLOB GPU_SOURCE_FILES src/*.cu)
    find_package(CUDA QUIET REQUIRED)

    #regarding this 'set':
    #the problem was that -I/usr/local/cuda/include goes before normal include_dirs: -Isrc/ Iinclude/
    #and that produce <<identifier "cuda_random" is undefined>> etc..
    #that set generates <<-I,>>
    set(CUDA_INCLUDE_DIRS ",")

    add_definitions(-DGPU)
    include_directories(/usr/local/cuda/include/)
    link_directories(/usr/local/cuda/lib64)

    set(complier_options --compiler-options " \" ${COMMON_FLAGS} \" ")
    set(CUDA_ARCH_FLAGS -gencode arch=compute_52,code=[sm_52,compute_52])
#            -gencode arch=compute_35,code=sm_35
#            -gencode arch=compute_50,code=[sm_50,compute_50]
#            -gencode arch=compute_52,code=[sm_52,compute_52])


    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; ${CUDA_ARCH_FLAGS} ${complier_options})


    cuda_add_library(darknet-gpu ${GPU_SOURCE_FILES})
    #    target_link_libraries(darknet-gpu cuda cudart cublas curand)

    set(LINKER_LIBS ${LINKER_LIBS} cuda cudart cublas curand darknet-gpu stdc++)
endif ()

#remove Joseph's abandoned files
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*src/compare.c")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*examples/dice.c")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*examples/swag.c")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*examples/voxel.c")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*examples/writing.c")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*examples/rnn_vid.c")

add_executable(darknet ${SOURCE_FILES})
target_link_libraries(darknet ${LINKER_LIBS})


#add_library(darknet STATIC ${SOURCE_FILES})
#target_compile_options(darknet-bin PRIVATE -lm -lsybdb -lcurl)


#add_executable(darknet-bin ${SOURCE_FILES} include/darknet.c)
#target_link_libraries(darknet-bin ${LINKER_LIBS})


link_directories(/usr/local/lib64)


#set_target_properties(darknet-bin PROPERTIES OUTPUT_NAME darknet)
#add_custom_target(darknet-bin)

