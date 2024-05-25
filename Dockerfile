FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    echo "tzdata tzdata/Areas select Etc" > /tmp/preseed.txt && \
    echo "tzdata tzdata/Zones/Etc select UTC" >> /tmp/preseed.txt && \
    debconf-set-selections /tmp/preseed.txt && \
    apt-get install -y --no-install-recommends \
        tzdata \
        build-essential \
        curl \
        ca-certificates \
        gnupg \
        pkg-config \
        libssl-dev \
        lsb-release \
        software-properties-common \
        clang \
        libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
    curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH=/root/.cargo/bin:$PATH

# home directory
WORKDIR /home

CMD ["bash"]
