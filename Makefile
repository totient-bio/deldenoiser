.PHONY: example docs test docker_image
.DEFAULT_GOAL := example
PROJECT := deldenoiser
GIT_HASH := $(shell git log -1 --format="%h")
DOCKER_IMAGE := deldenoiser:$(GIT_HASH)


example:
	cd ./example && bash run_command_line_tool.bash


docs:
	pip install -r requirements-dev.txt
	sphinx-apidoc -f -e -T -M -o docs/api $(PROJECT) $(PROJECT)/tests/* $(PROJECT)/conftest.py
	cd docs && make html
	open docs/_build/html/index.html


test:
	pip install -r requirements-dev.txt
	py.test -v --pep8 "$(PROJECT)" --cov "$(PROJECT)" --flake8 "$(PROJECT)"


docker_image:
	docker build -t $(DOCKER_IMAGE) .
