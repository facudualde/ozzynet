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

sync:
	@test -d .venv || python3 -m venv .venv
	@.venv/bin/pip install --upgrade pip --quiet > /dev/null
	@echo "Installing dependencies (this may take a minute)..."
	@.venv/bin/pip install -r requirements.txt --quiet > /dev/null
	@.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet > /dev/null
	@echo ""
	@echo "Editor venv ready at .venv/"
	@echo "To use it in your shell: source .venv/bin/activate"

clean:
	docker compose down --remove-orphans
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -f .env
	rm -rf .venv

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
