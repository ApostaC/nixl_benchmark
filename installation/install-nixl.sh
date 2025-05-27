#!/bin/bash

export LIBRARY_PATH=${HOME}/local/lib:${LIBRARY_PATH}
export CPATH=${HOME}/local/include:${CPATH}

git clone https://github.com/ai-dynamo/nixl.git
cd nixl

# Turn off disable_mooncake_backend
#sed -i '/disable_mooncake_backend/d' meson_options.txt
#echo "option('disable_mooncake_backend', type : 'boolean', value : false, description : 'disable mooncake backend')" >> meson_options.txt
#

rm -rf build && \
    mkdir build && \
    uv run meson setup build/ --prefix=/home/yihua/local && \
    cd build && \
    ninja && \
    ninja install && cd  ..

uv build --wheel
uv pip install dist/*.whl

echo "Please add the following lines to /etc/ld.so.conf.d/nixl.conf and then run 'sudo ldconfig':"
echo "======================================================"
echo "/home/yihua/local/lib/x86_64-linux-gnu" 
echo "/home/yihua/local/lib/x86_64-linux-gnu/plugins" 
echo "======================================================"

