# This is just a default value in case the env
# variable is absent.
ARG DOCKER_IMAGE=rocm/pytorch:latest
FROM ${DOCKER_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# This is just a default value in case the env
# variable is absent.
ARG VIDEO_GID=44
ARG RENDER_GID=992
RUN getent group ${VIDEO_GID}  >/dev/null || groupadd -g ${VIDEO_GID}  host_video; \
    getent group ${RENDER_GID} >/dev/null || groupadd -g ${RENDER_GID} host_render

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
