#ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.10-py3
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.09-py3
FROM ${FROM_IMAGE_NAME}

# Set working directory
WORKDIR /workspace/ssd

# Copy the model files
COPY . .

ENV GIT_SSL_NO_VERIFY=true
# Install python requirements
RUN pip install --no-cache-dir --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host pypi.ngc.nvidia.com --trusted-host developer.download.nvidia.com -r requirements.txt

ENV CUDNN_V8_API_ENABLED=1
ENV TORCH_CUDNN_V8_API_ENABLED=1
