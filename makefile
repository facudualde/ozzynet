.ONESHELL:

setup:
	echo "UID=$$(id -u)" > .env
	echo "GID=$$(id -g)" >> .env
	echo "VIDEO_GID=$$(getent group video | cut -d: -f3)" >> .env
	echo "RENDER_GID=$$(getent group render | cut -d: -f3)" >> .env
	echo "HIP_VISIBLE_DEVICES=0" >> .env
	echo "# Change this flag to trick Pytorch and detect your GPU, for example 10.3.0" >> .env
	echo "# HSA_OVERRIDE_GFX_VERSION=CHANGE_ME" >> .env
	echo "DOCKER_IMAGE=rocm/pytorch:latest" >> .env

	chmod u+x entrypoint.sh

clean:
	docker compose down --remove-orphans
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -f .env

up:
	docker compose up -d

down:
	docker compose down

shell:
	docker compose exec pytorch bash

health:
	docker compose exec pytorch python3 src/health.py

# TODO
# train:
	# docker compose exec pytorch python3 src/train.py
