# This is just a default value in case the env
# variable is absent.
ARG DOCKER_IMAGE=rocm/pytorch:latest
FROM ${DOCKER_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
