# First environment for the build process
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    zip \
    unzip \
    tar \
    pkg-config \
    libgomp1 \
    libeigen3-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY . .
RUN cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release

# Second environement just with the compiled version
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libhdf5-103 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sisap2026

# Pull the compiled binary
COPY --from=builder /src/build/main /app/chnsw_app

COPY ./config /sisap2026/config
COPY ./results /sisap2026/results

RUN mkdir -p /sisap2026/config /sisap2026/results

ENV OMP_NUM_THREADS=8
ENV OMP_PLACES=cores
ENV OMP_PROC_BIND=close

ENTRYPOINT ["/app/chnsw_app"]