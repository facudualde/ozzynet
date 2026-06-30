#!/bin/bash
set -e

# Create video and render grupos if they don't already exist in the container.
getent group "${VIDEO_GID}" >/dev/null || groupadd -g "${VIDEO_GID}" host_video 2>/dev/null
getent group "${RENDER_GID}" >/dev/null || groupadd -g "${RENDER_GID}" host_render 2>/dev/null

# Change to host user.
exec chroot --userspec="${UID}:${GID}" / env \
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
    HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION}" \
    PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
    TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}" \
    PATH="${PATH}" \
    HOME="/workspace" \
    "$@"
