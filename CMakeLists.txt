cmake_minimum_required(VERSION 3.18...4.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(sdefin)

# We only need pybind11 from pip / submodule
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(OpenMP)

find_library(BLAS_LIB gslcblas  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/libs/gsl/lib)
find_library(FFTW_LIB fftw3 PATHS ${CMAKE_CURRENT_SOURCE_DIR}/libs/fftw/lib)
find_library(GSL_LIB gsl PATHS ${CMAKE_CURRENT_SOURCE_DIR}/libs/gsl/lib)

# Build the Python extension module
python_add_library(sdefin MODULE bindings/bindings.cpp WITH_SOABI)

# Always use vendored headers
target_include_directories(sdefin
INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include/> 
    PRIVATE
        /usr/include/
        /usr/include/x86_64-linux-gnu
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/api
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/eigen
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/armadillo/include
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/boost
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/boost/libs/math/include 
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/fftw/include
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/gsl/include
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/pybind11/include
)



target_link_libraries(sdefin PRIVATE ${BLAS_LIB} ${FFTW_LIB} ${GSL_LIB} OpenMP::OpenMP_CXX pybind11::headers)
target_compile_definitions(sdefin PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

install(TARGETS sdefin DESTINATION  ${SKBUILD_PLATLIB_DIR})



