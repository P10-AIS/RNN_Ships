FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget bzip2 git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

# Accept ToS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

ENV CONDA_OVERRIDE_CUDA=11.2

WORKDIR /app

# Copy BOTH environment files
COPY model_fitting_environment.yml .
COPY processing_environment.yml .

# Build BOTH environments
RUN conda env create -f model_fitting_environment.yml && \
    conda env create -f processing_environment.yml && \
    conda clean -a -y

# Initialize conda for bash so VS Code can use `conda activate`
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# 🚨 DO NOT ADD 'ENV PATH=/opt/conda/envs/...'. Let Conda handle the path!

CMD ["python"]