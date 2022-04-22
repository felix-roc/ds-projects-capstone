SHELL := /bin/bash

.PHONY: setup data clean
## Setup the virtual environment and install requirements
setup:
	pyenv local 3.10.3
	python -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

## Make Dataset
data:
	python -m src.make_dataset

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
