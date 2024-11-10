# Use a base image with CUDA and Python
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    gcc-9 \
    g++-9 \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set environment variables
ENV CRYPTOGRAPHY_OPENSSL_NO_LEGACY=true \
    BUILD_WITH_CUDA=1 \
    CUDA_HOST_COMPILER="/usr/bin/gcc-9" \
    CUDA_HOME="/usr/local/cuda" \
    CUDA_PATH="/usr/local/cuda" \
    FORCE_CUDA=1 \
    MAX_JOBS=12 \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6" \
    AM_I_DOCKER=True

# Install Python packages
RUN pip install --upgrade pip
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
RUN pip install "mmsegmentation>=1.0.0"
RUN pip install ipykernel ftfy regex tqdm networkx==2.8.8
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx
RUN pip install scikit-image

RUN pip install torch_geometric
RUN pip install openai==0.28
RUN pip install git+https://github.com/openai/CLIP.git

WORKDIR /workspace
# Copy all necessary folders to /workspace in the container
COPY . .

# Change directory to install projectaria_tools
WORKDIR /workspace/projectaria_tools
RUN python3 -m pip install projectaria-tools'[all]'
WORKDIR /workspace
# Default command
CMD ["/bin/bash"]
