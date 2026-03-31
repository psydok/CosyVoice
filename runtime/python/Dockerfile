FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget build-essential sox libsox-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN uv python install 3.10 && uv venv /opt/venv --python 3.10
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

WORKDIR /opt/cosyvoice

RUN uv pip install vllm[transformers,cuda]==0.17.1 --no-cache

RUN uv pip install \
    tensorrt-cu12==10.13.3.9 \
    tensorrt-cu12-bindings==10.13.3.9 \
    tensorrt-cu12-libs==10.13.3.9 \
    --no-cache
RUN uv pip install "setuptools==81.0.0" -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN uv pip install -U openai-whisper  --no-deps --no-build-isolation

RUN git clone --recursive https://github.com/psydok/CosyVoice.git .
RUN sed -i -E 's/^(.*openai-whisper.*)$/# \1/' requirements.txt
RUN uv pip install -r requirements.txt --no-cache --index-strategy unsafe-best-match
RUN uv pip install -U numpy==1.26.4

ENV PYTHONPATH="/opt/cosyvoice:/opt/cosyvoice/third_party/Matcha-TTS"
RUN uv pip install \
    torch==2.10.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --no-cache

RUN uv pip install -U "transformers>=4.56,<5" numpy==1.26.4
ENV FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"
RUN uv pip install https://github.com/Dao-AILab/flash-attention.git --no-deps --no-build-isolation

RUN cd runtime/python/grpc && python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto
ENV VLLM_NO_USAGE_STATS=1
