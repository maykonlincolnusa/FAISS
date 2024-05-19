#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

#cp -r -n $CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include/* $CONDA_PREFIX/include/
#cp -r -n $CONDA_PREFIX/x86_64-conda-linux-gnu/include/c++/11.2.0/* $CONDA_PREFIX/include/
#cp -r -n $CONDA_PREFIX/x86_64-conda-linux-gnu/include/c++/11.2.0/x86_64-conda-linux-gnu/* $CONDA_PREFIX/include/

# Build libfaiss.so/libfaiss_avx2.so/libfaiss_avx512.so
cmake -B _build \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTING=OFF \
      -DFAISS_OPT_LEVEL=avx512 \
      -DFAISS_ENABLE_GPU=ON \
      -DFAISS_ENABLE_RAFT=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DBLA_VENDOR=Intel10_64lp \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
      -DCMAKE_BUILD_TYPE=Release .

make -C _build -j$(nproc) faiss faiss_avx2 faiss_avx512

cmake --install _build --prefix $PREFIX
cmake --install _build --prefix _libfaiss_stage/
