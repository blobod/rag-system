FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

# install python packages
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
# port for fastapi
EXPOSE 8000

CMD ["/bin/bash"]
