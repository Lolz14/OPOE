#!/usr/bin/env bash
set -e
ROOT=$(pwd)
LIBS=$ROOT/libs
mkdir -p $LIBS
cd $LIBS
echo "📦 Setting up local dependencies in $LIBS"
# Eigen
if [ ! -d eigen ]; then
  echo "➡️ Downloading Eigen..."
  curl -L -o eigen.tar.gz https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  tar -xzf eigen.tar.gz
  mv eigen-3.4.0 eigen
  rm eigen.tar.gz
fi
# Pybind11
if [ ! -d pybind11 ]; then
  echo "➡️ Downloading pybind11..."
  curl -L -o pybind11.tar.gz https://github.com/pybind/pybind11/archive/refs/heads/master.tar.gz
  tar -xzf pybind11.tar.gz
  mv pybind11-master pybind11
  rm pybind11.tar.gz
fi
# Boost (headers only)
if [ ! -d boost ]; then
  echo "➡️ Downloading Boost..."
  curl -L -o boost.tar.gz https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.gz
  tar -xzf boost.tar.gz
  mv boost-1.84.0 boost
  rm boost.tar.gz
fi
# Armadillo
if [ ! -d armadillo ]; then
  echo "➡️ Downloading Armadillo..."
  curl -L -o armadillo.tar.xz https://sourceforge.net/projects/arma/files/armadillo-12.8.3.tar.xz
  tar -xf armadillo.tar.xz
  mv armadillo-12.8.3 armadillo
  rm armadillo.tar.xz
fi
# FFTW
if [ ! -d fftw ]; then
  echo "➡️ Downloading FFTW..."
  curl -L -o fftw.tar.gz http://www.fftw.org/fftw-3.3.10.tar.gz
  tar -xzf fftw.tar.gz
  mv fftw-3.3.10 fftw
  cd fftw
  mkdir -p lib include
  ./configure --prefix=$LIBS/fftw --enable-shared --enable-threads
  make -j$(nproc)
  make install
  cd ..
  rm fftw.tar.gz
fi
# GSL
if [ ! -d gsl ]; then
  echo "➡️ Downloading GSL..."
  curl -L -o gsl.tar.gz ftp://ftp.gnu.org/gnu/gsl/gsl-2.7.1.tar.gz
  tar -xzf gsl.tar.gz
  mv gsl-2.7.1 gsl
  cd gsl
  mkdir -p lib include
  ./configure --prefix=$LIBS/gsl
  make -j$(nproc)
  make install
  cd ..
  rm gsl.tar.gz
fi
echo "✅ All dependencies installed locally in $LIBS"
