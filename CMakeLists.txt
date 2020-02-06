cmake_minimum_required(VERSION 3.10)
project(Algorithms)

set(CMAKE_CXX_STANDARD 17)

include(FindLAPACK)
find_package(Armadillo REQUIRED)
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(include)
include_directories(src)

set(CMAKE_CXX_FLAGS "-O3 -fopenmp -DARMA_DONT_USE_WRAPPER -DARMA_USE_ARPACK -DARMA_USE_HDF5")

add_executable(Algorithms
        include/OMF/FixedPenalty.h
        include/OMF/FixedTolerance.h
        include/LSRN/LatentSpaceRN.h
        include/MAME/MAME_svd.h
        include/OATS/OATS_ogd.h
        include/OTS/OTS_gsr.h
        include/OTS/OTS_ogd.h
        include/OMF/TemplateOMF.h
        include/OTS/TemplateOTS.h
        include/OMF/ZeroTolerance.h
        include/OATS/TemplateOATS.h
        include/TRMF/TRMF.h
        include/TRMF/tron.h
        include/BEATLEX/Beatlex.h
        src/BEATLEX/Beatlex.cpp
        src/TRMF/tron.cpp
        src/TRMF/TRMF.cpp
        src/OMF/FixedPenalty.cpp
        src/OMF/FixedTolerance.cpp
        src/LSRN/LatentSpaceRN.cpp
        src/main.cpp
        src/MAME/MAME_svd.cpp
        src/OATS/OATS_ogd.cpp
        src/OTS/OTS_gsr.cpp
        src/OTS/OTS_ogd.cpp
        src/OATS/TemplateOATS.cpp
        src/OMF/TemplateOMF.cpp
        src/OTS/TemplateOTS.cpp
        src/OMF/ZeroTolerance.cpp)

target_link_libraries(
        Algorithms
        ${LAPACK_LIBRARIES}
        ${Armadillo_LIBRARIES}
        -lblas
        -llapack
        ${HDF5_LIBRARIES}
)