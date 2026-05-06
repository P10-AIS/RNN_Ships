FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget bzip2 git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh && \
    bash Miniforge3-Linux-aarch64.sh -b -p /opt/conda && \
    rm Miniforge3-Linux-aarch64.sh

ENV PATH=/opt/conda/bin:$PATH

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