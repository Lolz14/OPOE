cmake_minimum_required(VERSION 3.18...4.0)

project(opoe VERSION 0.1.0 LANGUAGES CXX)

# C++ standard (portable)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release for single-config generators (Ninja/Makefiles)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Avoid forcing -O3 flags; use generator defaults for Release across compilers.
# If you still want O3 in Release, uncomment:
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options($<$<CONFIG:Release>:-O3>)
elseif(MSVC)
  add_compile_options($<$<CONFIG:Release>:/O2>)
  endif()

# Python + pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# OpenMP (optional; used if present)
find_package(OpenMP QUIET)

# Optional header-only/math libs (enable if your code uses them)
# Eigen3
find_package(Eigen3 3.3 QUIET NO_MODULE)

# Armadillo (templated headers + thin lib; optional)
find_package(Armadillo QUIET)

# Boost headers (for things like boost::math, header-only)
find_package(Boost QUIET)

# FFTW3 and GSL (required based on your original link line)
# Prefer CMake package configs; fall back to pkg-config if needed.
find_package(PkgConfig QUIET)

# GSL
# Try CMake config first (provides GSL::gsl and optionally GSL::gslcblas)
find_package(GSL CONFIG QUIET)
if(NOT TARGET GSL::gsl)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_GSL REQUIRED IMPORTED_TARGET gsl)
    add_library(GSL::gsl INTERFACE IMPORTED)
    target_link_libraries(GSL::gsl INTERFACE PkgConfig::PC_GSL) # usually includes gsl and gslcblas
  else()
    message(FATAL_ERROR "GSL not found. Install GSL (e.g., apt/brew/vcpkg/conda) or ensure pkg-config is available.")
  endif()
endif()

# FFTW3 (double-precision)
# Try CMake config first (FFTW3::fftw3)
find_package(FFTW3 CONFIG QUIET)
if(NOT TARGET FFTW3::fftw3)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_FFTW3 REQUIRED IMPORTED_TARGET fftw3)
    add_library(FFTW3::fftw3 INTERFACE IMPORTED)
    target_link_libraries(FFTW3::fftw3 INTERFACE PkgConfig::PC_FFTW3)
  else()
    message(FATAL_ERROR "FFTW3 not found. Install FFTW3 (e.g., apt/brew/vcpkg/conda) or ensure pkg-config is available.")
  endif()
endif()

# Build the Python extension module
pybind11_add_module(opoe MODULE bindings/bindings.cpp WITH_SOABI)

# Visibility (helps Python extension size/ABI)
set_target_properties(opoe PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# Project headers
target_include_directories(opoe
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_include_directories(opoe PRIVATE
        /usr/include
        /usr/include/x86_64-linux-gnu
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin") # macOS
    target_include_directories(opoe PRIVATE
        /usr/include
        /opt/homebrew/include  # Homebrew default
        /usr/local/include     # Intel mac default
    )
elseif(WIN32)
    target_include_directories(opoe PRIVATE
        "C:/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/<version>/include"
        "C:/Program Files (x86)/Windows Kits/10/Include/<sdk-version>/ucrt"
    )
endif()


# Link against imported targets only (portable)
target_link_libraries(opoe
  PRIVATE
    pybind11::headers
    FFTW3::fftw3
    GSL::gsl
)

# Link GSL CBLAS only if the target exists (some configs include it transitively)
if(TARGET GSL::gslcblas)
  target_link_libraries(opoe PRIVATE GSL::gslcblas)
endif()

# Optional dependencies: link to propagate include dirs if found
if(TARGET Eigen3::Eigen)
  target_link_libraries(opoe PRIVATE Eigen3::Eigen)
endif()

if(TARGET armadillo)
  target_link_libraries(opoe PRIVATE armadillo)
endif()

if(Boost_FOUND)
  if(TARGET Boost::headers)
    target_link_libraries(opoe PRIVATE Boost::headers)
  elseif(TARGET Boost::boost)
    target_link_libraries(opoe PRIVATE Boost::boost)
  endif()
endif()

# OpenMP if available
if(OpenMP_CXX_FOUND)
  target_link_libraries(opoe PRIVATE OpenMP::OpenMP_CXX)
endif()

# Version macro for your sources
target_compile_definitions(opoe PRIVATE VERSION_INFO="${PROJECT_VERSION}")

# Install into Python site-packages (for scikit-build or plain CMake)
include(GNUInstallDirs)
if(DEFINED SKBUILD_PLATLIB_DIR)
  set(_PY_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}")
elseif(DEFINED Python3_SITELIB)
  set(_PY_INSTALL_DIR "${Python3_SITELIB}")
else()
  set(_PY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
endif()

install(
  TARGETS opoe
  LIBRARY DESTINATION "${_PY_INSTALL_DIR}"
  RUNTIME DESTINATION "${_PY_INSTALL_DIR}"  # Windows .pyd
  ARCHIVE DESTINATION "${_PY_INSTALL_DIR}"  # edge cases
)