# Dockerfile for Face Analysis Model
# Alternative to Cog build for local testing

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    insightface==0.7.3 \
    onnxruntime-gpu==1.23.2 \
    deepface==0.0.95 \
    tensorflow==2.15.0 \
    tf-keras==2.15.0 \
    opencv-python-headless==4.10.0.84 \
    numpy==1.26.4 \
    scikit-learn==1.7.2 \
    scipy==1.16.3 \
    protobuf==3.20.3

# Copy prediction code
COPY predict.py /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port (if running as web service)
EXPOSE 5000

# Default command
CMD ["python", "predict.py"]
