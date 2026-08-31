# 1. 国内代理获取 uv
FROM ghcr.m.daocloud.io/astral-sh/uv:latest AS uv_bin

# 2. CUDA 基础镜像
FROM docker.m.daocloud.io/nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 3. 配置 apt 源 + 安装 Python 3.13
RUN \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common git curl ca-certificates gnupg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 && \
    echo "deb https://launchpad.proxy.ustclug.org/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes-ppa.list && \
    apt-get update && \
    apt-get install -y python3.13 python3.13-dev && \
    rm -rf /var/lib/apt/lists/*

# 注入 uv，设置 python 软链
COPY --from=uv_bin /uv /usr/local/bin/uv
RUN ln -sf /usr/bin/python3.13 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.13 /usr/bin/python

# 4. 安装 Python 依赖（显式指定清华源，禁用官方源回退）
COPY requirements.txt .
RUN uv pip install --system --no-cache \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

CMD ["/bin/bash"]