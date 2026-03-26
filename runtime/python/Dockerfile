FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends \
        software-properties-common \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip && \
    apt-get install -y \
        wget curl git nano ca-certificates sox libsox-dev git unzip git-lfs g++ curl && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /opt/CosyVoice
RUN pip install --no-cache-dir -U pip==22.0.2

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get -y install git unzip git-lfs g++
RUN git lfs install
RUN git clone --recursive https://github.com/psydok/CosyVoice.git
# here we use python==3.10 because we cannot find an image which have both python3.8 and torch2.0.1-cu118 installed
RUN cd CosyVoice && pip3 install -U -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com --no-cache-dir
RUN cd CosyVoice/runtime/python/grpc && python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto
# Не оптимизировано просто, чтобы запустился grpc с изменениями
RUN apt install -y cuda-toolkit-12-4
RUN pip install vllm==v0.11.0 transformers==4.57.1 numpy==1.26.4 onnxruntime-gpu==1.23.2
RUN pip install flashinfer-python flashinfer-cubin
RUN pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128
# тест flash-attn, возможно не закончен. 
# RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# Чтобы собрать изменения
# COPY . CosyVoice/runtime/python
# RUN mv /opt/CosyVoice/CosyVoice/cosyvoice /opt/CosyVoice/CosyVoice/cosyvoice_orig
