#!/bin/bash

set -xe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${HOME}/local/"
INSTALL_LIB_DIR="${INSTALL_DIR}/lib"

mkdir -p "${INSTALL_DIR}"

export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib:${INSTALL_LIB_DIR}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/local/lib:${INSTALL_LIB_DIR}

UCX_VERSION=v1.18.0
OPENMPI_VERSION=5.0.6


# Install GDR
git clone https://github.com/NVIDIA/gdrcopy.git
prefix=${INSTALL_DIR} make -C ./gdrcopy lib_install
sudo ldconfig ${INSTALL_LIB_DIR}

# Install UCX
curl -fSsL "https://github.com/openucx/ucx/tarball/${UCX_VERSION}" | tar xvz
cd openucx-ucx*
./autogen.sh && ./configure     \
    --prefix=${INSTALL_DIR}     \
    --enable-shared             \
    --disable-static            \
    --disable-doxygen-doc       \
    --enable-optimizations      \
    --enable-cma                \
    --enable-devel-headers      \
    --with-cuda=/usr/local/cuda \
    --with-verbs                \
    --with-dm                   \
    --with-gdrcopy=${INSTALL_DIR}   \
    --enable-mt

make -j24
make -j24 install-strip
sudo ldconfig ${INSTALL_LIB_DIR}

cd ${SCRIPT_DIR}

