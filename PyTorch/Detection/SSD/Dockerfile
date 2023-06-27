ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.09-py3
FROM ${FROM_IMAGE_NAME}

# Set working directory
WORKDIR /workspace/ssd

# Copy the model files
COPY . .

ENV GIT_SSL_NO_VERIFY=true
# Install python requirements
RUN pip install --no-cache-dir -r requirements.txt

ENV CUDNN_V8_API_ENABLED=1
ENV TORCH_CUDNN_V8_API_ENABLED=1
